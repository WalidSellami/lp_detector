import json
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from fastapi.responses import JSONResponse
import requests
from sqlalchemy.orm import Session
import app.database as database
import app.models as models
from pydantic import BaseModel
from passlib.context import CryptContext

import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
# from PIL import Image
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "license_detector_model.pt")

model = YOLO(model_path)

# Load OCR
ocr = PaddleOCR(lang='en')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Manage connected clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: dict):
        message_text = json.dumps(message)
        for connection in self.active_connections:
            await connection.send_text(message_text)

manager = ConnectionManager()

@app.websocket("/ws/notifications")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)



class LicensePlateCreate(BaseModel):
    image: str
    plate_number: str
    
class LicensePlateSearchRequest(BaseModel):
    plate_number: str    
    

class AlertData(BaseModel):
    status: str
    vehicle_status: str
    plate_number: str
      
    
class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    
class UserLogin(BaseModel):
    email: str
    password: str    
    
       
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(database.get_db)):
    if db.query(models.Users).filter((models.Users.email == user.email) | (models.Users.name == user.name)).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email or name already registered")
    
    new_user = models.Users(
        name=user.name,
        email=user.email,
        password_hash=hash_password(user.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully", "user_id": new_user.id}


@app.post("/login")
def login(user: UserLogin, db: Session = Depends(database.get_db)):
    db_user = db.query(models.Users).filter(models.Users.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
       return {"status": "fail", "message": "Invalid email or password", "user_id": None}
    
    return {"status": "success", "message": "Sign in successful", "user_id": db_user.id}


@app.get('/profile/{user_id}')
def get_profile(user_id: int, db: Session = Depends(database.get_db)):
    db_user = db.query(models.Users).filter(models.Users.id == user_id).first()
    if not db_user:
        return {'User not found'}
    
    return {"user_id": db_user.id, "name": db_user.name, "email": db_user.email}    


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    url = "https://freeimage.host/api/1/upload"
    api_key = "" # get it from https://freeimage.host/
    
    file_contents = await file.read()
    
    params = {"key": api_key}
    files = {"source": (file.filename, file_contents)}
    
    response = requests.post(url, data=params, files=files)

    if response.status_code == 200:
        result = response.json()
        return JSONResponse(content={"image_url": result["image"]["url"]})
    else:
        return JSONResponse(status_code=400, content={"error": "Image upload failed"})       

@app.post('/add_license_plate')
def create_license_plate(license_plate: LicensePlateCreate, db: Session = Depends(database.get_db)):
    try:
        db_license_plate = models.LicensePlate(image=license_plate.image, 
                                               plate_number=license_plate.plate_number)
        
        db.add(db_license_plate)
        db.commit()
        db.refresh(db_license_plate)
        return db_license_plate
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put('/edit_license_plate/{plate_id}')
def edit_license_plate(plate_id: int, license_plate: LicensePlateCreate, db: Session = Depends(database.get_db)):
    try:
        db_license_plate = db.query(models.LicensePlate).filter(models.LicensePlate.id == plate_id).first()
        if db_license_plate is None:
            raise HTTPException(status_code=404, detail="License plate not found")
        db_license_plate.image = license_plate.image
        db_license_plate.plate_number = license_plate.plate_number
        db.commit()
        db.refresh(db_license_plate)
        return db_license_plate
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/delete_license_plate/{plate_id}')
def delete_license_plate(plate_id: int, db: Session = Depends(database.get_db)):
    db_license_plate = db.query(models.LicensePlate).filter(models.LicensePlate.id == plate_id).first()
    if db_license_plate is None:
        raise HTTPException(status_code=404, detail="License plate not found")
    db.delete(db_license_plate)
    db.commit()
    return {"message": "License plate deleted successfully"}


@app.get('/license_plates/{plt_numb}')
def get_license_plate(plt_numb: int, db: Session = Depends(database.get_db)):
    license_plate = db.query(models.LicensePlate).filter(models.LicensePlate.plate_number == plt_numb).first()
    if license_plate is None:
        raise HTTPException(status_code=404, detail="License plate not found")
    return license_plate


@app.get('/license_plates')
def get_license_plates(db: Session = Depends(database.get_db)):
    try:
       
        license_plates = db.query(models.LicensePlate).all()
        
        license_plates_list = [lp.to_dict() for lp in license_plates]
        
        content = {
            'status': 'success',
            'license_plates': license_plates_list
        }
        
        return JSONResponse(content=content)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post('/search_license_plate')
def search_license_plate(request: LicensePlateSearchRequest, db: Session = Depends(database.get_db)):
  
    plate_number = request.plate_number

    license_plates = db.query(models.LicensePlate).filter(models.LicensePlate.plate_number == plate_number).all()
    
    if license_plates is None:
        content = {
            'status': 'success',
            'license_plates': []
        }
        return JSONResponse(content=content) 
    
    else:
        
        license_plates_list = [lp.to_dict() for lp in license_plates]
        
        content = {
           'status': 'success',
           'license_plates': license_plates_list
        }

        return JSONResponse(content=content)  


# @app.post("/upload_image")
# async def detect_license_plate(file: UploadFile = File(...)):
  
#     filename = file.filename
     
#     # check if the filename contains special characters
#     # if any(not c.isascii() for c in filename):
#     #     # generate random filename
#     #     filename = ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + f'.{filename.split(".")[-1]}'
            
#     image_path = f"E:\\All_Projects\\DL_ML_Projects\\license_plate_detection\\uploads\\{filename}"
 
#     os.makedirs(os.path.dirname(image_path), exist_ok=True) 

#     # Save the file asynchronously
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
        
    
#     try:
#         # Open the image using PIL
#         image = Image.open(image_path)

#         # For now, we'll return the file path as a response
#         return JSONResponse(content={"message": "Image processed successfully", "image_path": image_path})

#     except Exception as e:
#         # If there's an error opening the image, return an error response
#         return JSONResponse(content={"error": "Failed to process image", "details": str(e)}, status_code=400)      


# @app.post("/send_notification")
# async def send_notification(notification: NotificationData):
#     try:
#         title = notification.title
#         message = notification.message
#         await manager.send_message({
#             "title": title,
#             "message": message
#         })
#         return {
#             "message": "Notification sent successfully"
#         }
#     except Exception as e:
#         return JSONResponse(content={"error": "Failed to send notification", "details": str(e)}, status_code=500)


import time

# Dictionary to track the last time a message was sent for each license plate
license_plate_last_sent = {}

# Cooldown period (in seconds)
COOLDOWN_PERIOD = 15

@app.post("/detect_license_plate")
async def detect_license_plate(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    
    filename = file.filename
            
    image_path = f"E:\\All_Projects\\DL_ML_Projects\\license_plate_detection\\uploads\\{filename}"
 
    os.makedirs(os.path.dirname(image_path), exist_ok=True) 

    # Save the file asynchronously
    with open(image_path, "wb") as f:
        f.write(await file.read())
        
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   

    # Run YOLO model on the frame
    results = model(img)
    results_boxes = results[0].boxes.data.tolist() 

    detections = []
    
    if results_boxes != []:
        for result in results:
           for box in result.boxes:
              x1, y1, x2, y2 = map(int, box.xyxy[0])
              conf = box.conf[0]
              
              print(f'Confidence: {conf}')
  
              if conf >= 0.35:  # Confidence threshold
                  # Crop the license plate region
                  license_plate_image = img[y1:y2, x1:x2]
                  gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
                  filtered = cv2.bilateralFilter(gray, 3, 25, 25)
                  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                  enhanced_image = clahe.apply(filtered)
  
                  # OCR to recognize text
                  result_ocr = ocr.ocr(enhanced_image, det=True, rec=True)
  
                  license_plate_number = ''
                  if result_ocr != [None]:
                      for line in result_ocr:
                          for element in line:
                              license_plate_number = element[1][0]
                              license_plate_number = license_plate_number.replace(' ', '')
                              
                              print(f'License plate number: {license_plate_number}')
                              
                              
                              status_vehicle = ''
                              
                              # check if the license plate number in database
                              license_plate = db.query(models.LicensePlate).filter(models.LicensePlate.plate_number == license_plate_number).first()
                              if license_plate is not None:
                                  status_vehicle = 'Authorized'
                                #   # send notification
                                #   await manager.send_message({
                                #       "title": "Authorized",
                                #       "message": f"License plate number: {license_plate_number}"
                                #   })
                                  
                              else:
                                  status_vehicle = 'Unauthorized'
                                  current_time = time.time()

                                  if license_plate_number in license_plate_last_sent:
                                    last_sent_time = license_plate_last_sent[license_plate_number]  
                                    if (current_time - last_sent_time) > COOLDOWN_PERIOD:
                                        await manager.send_message({
                                            "title": "Unauthorized",
                                            "message": f"License plate number: {license_plate_number}",
                                            "plate_number": license_plate_number
                                        })

                                        # Update last sent time
                                        license_plate_last_sent[license_plate_number] = current_time    
                                    
                                  else:
                                      
                                        # Update last sent time
                                        license_plate_last_sent[license_plate_number] = current_time    
                                    
                                  # add alert 
                                  try:
                                    db_alert = models.Alerts(status='Unread', vehicle_status=status_vehicle, plate_number=license_plate_number)
                                    db.add(db_alert)
                                    db.commit()
                                    db.refresh(db_alert)
                                  except Exception as e:
                                      raise HTTPException(status_code=500, detail=str(e))
                                  
                                #   print(license_plate_last_sent)
                                            
                           
                              # Collect bounding box and license plate number
                              detections.append({
                                  'x': float(x1),
                                  'y': float(y1),
                                  'status': status_vehicle,
                                  'width': float(x2 - x1),
                                  'height': float(y2 - y1),
                                  'label': license_plate_number
                              })
                              
                              
                            #   print(detections)

    return JSONResponse(content={"detections": detections})



@app.post("/add_alert")
def create_alert(alert: AlertData, db: Session = Depends(database.get_db)):
    try:
        db_alert = models.Alerts(status=alert.status, vehicle_status=alert.vehicle_status, plate_number=alert.plate_number)
        db.add(db_alert)
        db.commit()
        db.refresh(db_alert)
        return db_alert
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/alerts")
def get_alerts(db: Session = Depends(database.get_db)):
    try:
        alerts = db.query(models.Alerts).order_by(models.Alerts.date.desc()).all()
        
        content = {
            'status': 'success',
            'alerts': [alert.to_dict() for alert in alerts]
        }
        
        return JSONResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  
    
    
@app.put("/edit_alerts")
def edit_alerts(db: Session = Depends(database.get_db)):
    try:
        db.query(models.Alerts).update({"status": "Read"})
        db.commit()
        # db.refresh(db.query(models.Alerts).all())
        
        return {"message": "All alerts are readed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
            
    
    
@app.delete("/delete_alert/{alert_id}")
def delete_alert(alert_id: int, db: Session = Depends(database.get_db)):
    try:
        db_alert = db.query(models.Alerts).filter(models.Alerts.id == alert_id).first()
        
        if db_alert is None:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        db.delete(db_alert)
        db.commit()
        
        return {"message": "Alert deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
    
  
  
@app.delete("/delete_all_alerts")
def delete_all_alerts(db: Session = Depends(database.get_db)):
    try:
        db.query(models.Alerts).delete()
        db.commit()
        
        return {"message": "All alerts deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))         


@app.get('/')
async def welcome():
    return {
        'message': 'Welcome to license plates detection API'
    }