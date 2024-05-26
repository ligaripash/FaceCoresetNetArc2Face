import os

import PIL.Image
from mtcnn import MTCNN
import cv2
import argparse


def detect_faces(input_dir, output_dir):
    # Initialize the MTCNN detector
    detector = MTCNN(steps_threshold=[0.6,0.7,0.9])


    # Walk through the input directory
    for root, _, files in os.walk(input_dir):

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                input_image_path = os.path.join(root, file)
                rel_path = os.path.relpath(os.path.dirname(input_image_path), input_dir)
                # Create the corresponding output directory structure
                #rel_path = os.path.relpath(input_image_path, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                print('computing ' + output_subdir)
                os.makedirs(output_subdir, exist_ok=True)

                # Load the input image
                image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
                #image = Image.open(input_image_path)
                #pixels = image.convert('RGB')
                #pixels = pixels.resize((224, 224))

                # Detect faces in the image
                faces = detector.detect_faces(image)
                image = PIL.Image.fromarray(image)
                # Save the detector output to a file
                faces_list = []
                for i, face in enumerate(faces):
                    face['id'] = i
                    x, y, width, height = face['box']
                    confidence = face['confidence']

                    output_filename = os.path.join(output_subdir, f'{file[0:-4]}_face_{i:04d}.png')
                    # Crop the face from the original image
                    face_image = image.crop((x, y, x + width, y + height))
                    face_image.save(output_filename, 'PNG')
                    face_list = [i, confidence, x, y, width, height ]
                    s = ','.join(str(x) for x in face_list) + '\n'
                    faces_list.append(s)


                faces_text_file_path = os.path.join(output_subdir, f'{file[0:-4]}.txt')
                with open(faces_text_file_path, 'w', newline='') as csv_file:
                    csv_file.writelines(faces_list)
                    #writer.writerow(faces.values())





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define the two arguments
    parser.add_argument('--input_directory', help="The root directory containing video frames")
    parser.add_argument('--output_directory', help="Where to place the output")

    input_directory = 'D:/faces/queries/'
    output_directory = 'D:/faces/queries_after_face_detection/'

    detect_faces(input_directory, output_directory)








