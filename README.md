# Online Attendance using Face Recognition
- Flask based web application to take attendance using face recognition.
- Provide two types of accounts: Students and Faculty.
- Faculties can train face recognition model on added photos by students, add lectures, take attendance by turn on camera, see and download the attendance, etc.
- Students can add their photos and see their attendance.
- This [articles](https://karm216.github.io/Fastpages-Notebooks/fastpages/jupyter/2021/03/25/face_recognization.html) shows Face recognition implementation and results of this project.
- Compatible Python Version = 3.6.9

## Steps to run this project:
### 1. Create Python virtual environment and install requirements
- Create virtual environment
```
python3.6 -m venv attendance-env
```
- Activate virtual environment
```
source attendance-env/bin/activate
```
- Upgrade Pip (because many a times pip version installed in old version (9.0.1), which sholud upgraded to latest version (20+))
 ```
 pip install --upgrade pip
 ``` 
- Install necessary requirements/libraries
```
pip install -r requirements.txt
```
### 2. Run Flask application
- Run flask app
```
python3 run.py
```
This will take 20-30 seconds to load pretrained models and finally give localhost url to run flask application. Then open that url (i.e. - `http://127.0.0.1:5500/`). 
By clicking url following home screen should be displayed.
<br></br>
![alt text](/Home.png?raw=true)
