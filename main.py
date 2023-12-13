import numpy as np
import cv2

from model import my_model, loading_trained_model


def main():

    print("loading model")
    model = my_model()
    # loading_trained_model(model)
    print("model loaded")

    haarcascade = cv2.CascadeClassifier(
        "haarcascade/haarcascade_frontalface_default.xml")
    # define a video capture object

    camera = cv2.VideoCapture(0)

    while (True):
        print("1")
        ret, frame = camera.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("2")

        detected_face = haarcascade.detectMultiScale(gray_frame, 1.3, 5)
        print("3")

        for (x, y, w, h) in detected_face:
            cv2.rectangle(gray_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray_frame[y:y+h, x:x+w]
            # roi_color = frame[y:y+h, x:x+w]
            img_copy = np.copy(gray_frame)

            original_width = roi_gray.shape[1]
            original_height = roi_gray.shape[0]

            img_gray = cv2.resize(roi_gray, (96, 96))
            img_gray = img_gray/255

            model_prediction_image = np.reshape(img_gray, (1, 96, 96, 1))
            predicted_keypoints = model.predict(model_prediction_image)[0]

            x_coordinate = predicted_keypoints[0::2]
            y_coordinate = predicted_keypoints[1::2]

            x_coordinate_denormalized = (x_coordinate + 0.5) * original_width
            y_coordinate_denormalized = (y_coordinate + 0.5) * original_height

            # 10 is on nose

            for i in range(len(x_coordinate_denormalized)):

                center = (int(x_coordinate_denormalized[i]), int(
                    y_coordinate_denormalized[i]))
                cv2.circle(roi_gray, center, 5, (0, 0, 0), -1)

            # Particular keypoints for scaling and positioning of the filter
            left_lip_coordinates = (int(x_coordinate_denormalized[11]), int(
                y_coordinate_denormalized[11]))
            right_lip_coordinates = (
                int(x_coordinate_denormalized[12]), int(y_coordinate_denormalized[12]))
            top_lip_coordinates = (int(x_coordinate_denormalized[13]), int(
                y_coordinate_denormalized[13]))
            bottom_lip_coordinates = (
                int(x_coordinate_denormalized[14]), int(y_coordinate_denormalized[14]))
            left_eye_coordinates = (
                int(x_coordinate_denormalized[3]), int(y_coordinate_denormalized[3]))
            right_eye_coordinates = (
                int(x_coordinate_denormalized[5]), int(y_coordinate_denormalized[5]))
            brow_coordinates = (int(x_coordinate_denormalized[6]), int(
                y_coordinate_denormalized[6]))

            # Scale filter according to keypoint coordinates
            beard_width = right_lip_coordinates[0] - left_lip_coordinates[0]
            glasses_width = right_eye_coordinates[0] - left_eye_coordinates[0]

            # Used for transparency overlay of filter using the alpha channel
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2BGRA)

            # sunglass filter
            glasses = cv2.imread('./face-filters/sunglass.png', -1)
            glasses = cv2.resize(glasses, (glasses_width*2, 150))
            gw, gh, gc = glasses.shape

            for i in range(0, gw):       # Overlay the filter based on the alpha channel
                for j in range(0, gh):
                    if glasses[i, j][3] != 0:
                        img_copy[brow_coordinates[1]+i+y-50,
                                 left_eye_coordinates[0]+j+x-60] = glasses[i, j]

            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGRA2BGR)
            cv2.imshow('Output', img_copy)

        cv2.imshow('Webcam', gray_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    camera.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
