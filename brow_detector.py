import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=0.5, circle_radius=1)
cap = cv2.VideoCapture(1)

text = ''

# r = 0.18533697614631156
# f_d = 0.4210272218051749

landmarks_l = [70,46,63,53,105,52,66,65,107,55]
landmarks_r = [336,285,296,295,334,282,293,283,300,276]

font                   = cv2.FONT_HERSHEY_SIMPLEX

bottomLeftCornerOfText = (400,400)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

buffer = 0
dist_list = []

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:

      a = results.multi_face_landmarks[0].landmark
      d = []
      f_d_new = (np.abs(a[10].x-a[152].x),np.abs(a[10].y-a[152].y),np.abs(a[10].z-a[152].z))
      f_d_new = np.sqrt(f_d_new[0]**2 + f_d_new[1]**2 + f_d_new[2]**2)
      
      for l in landmarks_l:
         d.append([np.abs(a[159].x-a[l].x),np.abs(a[159].y-a[l].y),np.abs(a[159].z-a[l].z)])
      for l in landmarks_r:
         d.append([np.abs(a[386].x-a[l].x),np.abs(a[386].y-a[l].y),np.abs(a[386].z-a[l].z)])

      d_dist = np.array(d).mean(0)
      r_new = np.sqrt(d_dist[0]**2 + d_dist[1]**2 + d_dist[2]**2)

      dist_list.append(r_new/f_d_new)
      
      buffer += 1
      if buffer > 100:
          dist_list.pop(0)
          buffer -= 1

      dist = np.array(dist_list).mean()

      if (r_new/f_d_new) > dist + 0.005/f_d_new :
          text = 'UP'
      elif (r_new/f_d_new) < dist - 0.003/f_d_new :
          text = 'DOWN'
      else:
          text = ''

      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
    # Flip the image horizontally for a selfie-view display.


    image = cv2.flip(image, 1)
    cv2.putText(image,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)    

    cv2.imshow('MediaPipe Face Mesh', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

