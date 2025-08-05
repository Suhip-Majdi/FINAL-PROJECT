import cv2
from imutils.video import WebcamVideoStream
import torch
import numpy as np
import pathlib
from selenium import webdriver

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class VideoCamera(object):
    def __init__(self):
        self.stream = WebcamVideoStream().start()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='new_10f.pt')
        self.model.eval()

        self.object_name = None

    def __del__(self):
        self.stream.stop()

    def get_frame_2(self):
        image = self.stream.read()

        #from BGR to RGB
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (640, 480))

        results = self.model(frame_rgb)

        print(f"results : {results}")

        # threshold
        threshold = 0.35
        filtered_results = []
        for i in range(len(results.pred[0])):
            if results.pred[0][i, 4] > threshold:
                filtered_results.append(results.pred[0][i])

        # update the results with threshold detections
        results.pred[0] = torch.stack(filtered_results) if filtered_results else torch.tensor([])

        # render the results bounding boxes and labels on the frame
        img_with_bboxes = results.render()[0]

        # back to BGR
        img_with_bboxes_bgr = cv2.cvtColor(img_with_bboxes, cv2.COLOR_RGB2BGR)

        # Encode the frame into JPEG format  /Frontend Compatibility
        ret, jpeg = cv2.imencode('.jpg', img_with_bboxes_bgr)
        data = [jpeg.tobytes()]

        return data

    def process_image(self, image):
        """
        This method processes an uploaded image for YOLOv5 detection and returns
        the processed image with bounding boxes for the highest confidence detection
        above the threshold of 20%.
        """
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # run YOLOv5 detection
        results = self.model(frame_rgb)

        # If there are no detections, return the original image
        if results.pred[0].shape[0] == 0:
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes(), {}

        threshold = 0.20
        conf_mask = results.pred[0][:, 4] > threshold
        filtered_results = results.pred[0][conf_mask]

        # If no detection meets the confidence threshold, return the original image
        if filtered_results.shape[0] == 0:
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes(), {}

        # get the detection with the maximum confidence score from the filtered results
        max_conf_idx = torch.argmax(filtered_results[:, 4])
        max_conf_detection = filtered_results[max_conf_idx].unsqueeze(0)

        # extract the class index of the detected object
        class_idx = int(max_conf_detection[0, 5])  # The 6th column is the class index
        self.object_name = self.model.names[class_idx]  # Get the name of the object
        print(f"Detected Object: {self.object_name}")

        # Prepare an object with the detection details (DIY and price URLs)
        detection_info = {
            "object_name": self.object_name,
            "diy_url": f"https://www.youtube.com/results?search_query=how+to+fix+or+replace+car+{self.object_name}",
            "price_url": f"https://www.amazon.com/s?k=car+{self.object_name}"
        }

        # Keep only the detection with the maximum confidence
        results.pred[0] = max_conf_detection

        # Render the results (bounding boxes and labels) on the frame
        img_with_bboxes = results.render()[0]

        # back to BGR for OpenCV compatibility
        img_with_bboxes_bgr = cv2.cvtColor(img_with_bboxes, cv2.COLOR_RGB2BGR)

        # Encode the image into JPEG format
        ret, jpeg = cv2.imencode('.jpg', img_with_bboxes_bgr)

        # Return both the processed image (JPEG bytes) and the detection info
        return jpeg.tobytes(), detection_info

    def diy(self):
        URL = f"https://www.youtube.com/results?search_query=how+to+fix+or+replace+car+{self.object_name}"
        print(URL)
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option("detach", True)
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(URL)

    def object_price(self):
        URL = f"https://www.amazon.com/s?k=car+{self.object_name}"
        print(URL)


#     def show(self):
#         while True:
#             frame = self.get_frame_2()[0]
#             np_img = np.frombuffer(frame, dtype=np.uint8)
#             image = cv2.imdecode(np_img, 1)
#
#             # Resize the frame for display
#             resized_frame = cv2.resize(image, (640, 480))
#             cv2.imshow('YOLOv5 Detection', resized_frame)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     cv2.destroyAllWindows()
#
#
# ob = VideoCamera()
# ob.show()
#     def diy(self):
#         URL = f"https://www.youtube.com/results?search_query=how+to+fix+or+replace+car+{self.object_name}"
#         print(URL)
#         # Keep Chrome browser open after program finishes
#         chrome_options = webdriver.ChromeOptions()
#         chrome_options.add_experimental_option("detach", True)
#
#         # Create and configure the Chrome webdriver
#         driver = webdriver.Chrome(options=chrome_options)
#
#         # Navigate to the (fake) newsletter registration page
#         driver.get(f"https://www.youtube.com/results?search_query=how+to+fix+or+replace+car+{self.object_name}")
#
#     def object_price(self):
#
#         URL = f"https://www.amazon.com/s?k=car{self.object_name}&crid=5JU59O59EITJ&sprefix=car+{self.object_name}+%2Caps%2C414&ref=nb_sb_noss_2"
#         print(URL)
#
#         # # Set up Chrome browser options
#         # chrome_options = webdriver.ChromeOptions()
#         # chrome_options.add_experimental_option("detach", True)
#         #
#         # # Create and configure the Chrome webdriver
#         # driver = webdriver.Chrome(options=chrome_options)
#         #
#         # # Navigate to the Amazon search URL
#         # driver.get(URL)  # Pass the URL, not the HTML content