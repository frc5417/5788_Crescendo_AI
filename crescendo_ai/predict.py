import cv2
from ultralytics import YOLO
import pathlib
from .log_init import MainLogger


class YoLoV8_Inference:
    def __init__(self, model_pt):
        self.model_path = str(pathlib.Path(model_pt).resolve())
        self.model = YOLO(self.model_path)
        root_log = MainLogger()
        self.log = root_log.StandardLogger("YoLoV8_Inference")
        self.log.info("Initializing YOLOv8 inference")
        self.log.info("Model loaded from: " + self.model_path)

    def inference_stream(self, stream):
        try:
            stream = int(stream)
            self.log.info("Inference on webcam: " + str(stream))
        except ValueError:
            self.log.info("Inference on video: " + stream)
        cap = cv2.VideoCapture(stream)

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 inference on the frame
                results = self.model(frame)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                #self.log.info(f"Preprocess Speed: {results[0].speed[0]}; Inference Speed: {results[0].speed[1]}; Postprocess Speed: {results[0].speed[2]}")
                #self.log.info(f"Classes: {results[0].names}; Confidence: {results[0].scores}")
                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
