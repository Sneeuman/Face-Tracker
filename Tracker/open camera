import cv2
from ultralytics import YOLO
from deepface import DeepFace

# -------------------------------
# Load YOLOv8 model
# -------------------------------
model= YOLO("yolov8n.pt")


# Create a VideoCapture object, 0 for default camera (change to 1, 2, etc. for external cameras)
cap = cv2.VideoCapture(0)

# Make the camera resoltion bigger
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

# Cheks if camera opened or not
if not cap.isOpened():
    print('Error: Could not open video stream or file')
else:
    print('Camara opened succefully. Press Escape to exit ')
    
living_thigs =[
    "person", "dog", "cat", "bird", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
]
    
while True:
    
    # Read a frame from the camera
    ret , frame = cap.read()
    
    if ret:

        
        #Run Yolo model on current frame
        results = model(frame ,conf=0.3)
        
        #loop throught results 
        for result in results:
            
            #Get all deteceted boxes 
            boxes = result.boxes
            
            #Loop through eatch deteced boxed
            for box in boxes:
                
                #Get the coordinates of the box
                x1, y1, x2, y2 = box.xyxy[0]
                
                #Converts Coordinates to integers
                x1, y1, x2, y2 = int(x1) , int(y1) , int(x2) , int(y2)
                
                #Get class ID (what object it detected)
                class_id = int(box.cls[0])
                
                #Get the class name(Example person , car , dog )
                class_name = model.names[class_id]
                
                
                #If the detected object 
                if class_name in living_thigs:
                    
                    color = (255,0,0)
                    display_name = "identity : " + class_name
                    
                    if class_name.lower() == "person":
                        person_img = frame[y1:y2,x1:x2]
                        try:
                            analysis = DeepFace.analyze(person_img, actoins = ['age'],enforce_detection = True)
                            age_text = f"age :  {int(analysis['age'])} "
                            
                        except:
                            age_text = "age : unknown"
                            
                        
                    
                    #draw a regtangle around the detected object 
                    cv2.rectangle(frame,(x1,y1), (x2, y2),color,2)
                    
                    #Put text above the rectangle
                    cv2.putText(
                        frame,
                        display_name,
                        (x1,y1 - 30), # Postion slightly above the box
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2 
                    )
                                
                    #Put text above the rectangle
                    cv2.putText(
                        frame,
                        age_text,
                        (x1,y1 - 10), # Postion slightly above the box
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2 
                    )
      
                else:
                    color = (0,255,0)
                    
                    display_name = "identity : " + class_name
                    
                    #draw a regtangle around the detected object 
                    cv2.rectangle(frame,(x1,y1), (x2, y2),color,2)
                    
                    #Put text above the rectangle
                    cv2.putText(
                        frame,
                        display_name,
                        (x1,y1 - 30), # Postion slightly above the box
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2 
                    )
                    
        
        #Rezise Window displayer to be larger
        resized_frame = cv2.resize(frame,(1200,720))
            
        # Display the frame in a window named "Webcam Feed"
        cv2.imshow('Webcam feed', resized_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Break the loop if the 'Esc' key (ASCII 27) is pressed       
    else:
        print("Error: Failed to capture frame. Exiting")
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()



