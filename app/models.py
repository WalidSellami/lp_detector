from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
import app.database as database


class Users(database.Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    

class LicensePlate(database.Base):
    __tablename__ = "license_plate"

    id = Column(Integer, primary_key=True, index=True)
    image = Column(String(500), unique=False, index=True)
    plate_number = Column(String(50), unique=False, index=True)
    date = Column(DateTime(timezone=True), server_default=func.now())
    
    def to_dict(self):
        date_str = self.date.isoformat() if self.date else None
        return {
            "id": self.id,
            "image": self.image,
            "plate_number": self.plate_number,
            "date": date_str,
        }
        
        
class Alerts(database.Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)  
    status = Column(String(50), unique=False, index=True)
    vehicle_status = Column(String(50), unique=False, index=True)
    plate_number = Column(String(100), unique=False, index=True)
    date = Column(DateTime(timezone=True), server_default=func.now())  
    
    
    def to_dict(self):
        date_str = self.date.isoformat() if self.date else None
        return {
            "id": self.id,
            "status": self.status,
            "vehicle_status": self.vehicle_status,
            "plate_number": self.plate_number,
            "date": date_str,
        }      
        
        
