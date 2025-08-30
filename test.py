###################### Tests ######################


# import pymysql

# try:
#     connection = pymysql.connect(host='127.0.0.1', user='root', password='ws@sm001', db='license_plates')
#     print("Connection successful")
# except pymysql.MySQLError as e:
#     print("Error connecting:", e)
# finally:
#     if connection:
#         connection.close()


import requests

def upload_image(file_path, api_key):
    
    url = "https://freeimage.host/api/1/upload"
    params = {"key": api_key}
    files = {"source": open(file_path, "rb")}  # Open the file in binary mode

    response = requests.post(url, data=params, files=files)

    if response.status_code == 200:
        result = response.json()
        print("Image URL:", result["image"]["url"])
        return result["image"]["url"]
    else:
        print("Error uploading image:", response.text)
        return None

# Example usage
image_path = "E:\\All_Projects\\DL_ML_Projects\\license_plate_detection\\Capture-d’écran-2024.png"
api_key = "6d207e02198a847aa98d0a2a901485a5"

upload_image(image_path, api_key)


# Convert to dart
# Future<void> uploadFile(String apiKey, String filePath) async {
#   final String url = "https://freeimage.host/api/1/upload";
#   final Dio dio = Dio();

#   try {
#     FormData formData = FormData.fromMap({
#       'key': apiKey,
#       'source': await MultipartFile.fromFile(filePath, filename: filePath.split('/').last),
#     });

#     Response response = await dio.post(url, data: formData);

#     if (response.statusCode == 200) {
#       print('File uploaded successfully: ${response.data}');
#     } else {
#       print('Failed to upload file: ${response.statusCode}');
#     }
#   } catch (e) {
#     print('Error occurred: $e');
#   }
# }