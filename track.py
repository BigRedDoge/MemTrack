from tracker import Tracker
import cv2


def main():
    tracker = Tracker(
        embedding_model_name='google/vit-base-patch16-224-in21k',
        yolo_model='yolov8m.pt',
        num_tables=10,
        device='cuda',
        verbose=True
    )

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = tracker.track(frame)
        if results:
            results.plot()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.stop()



if __name__ == "__main__":
    main()