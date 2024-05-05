from tracker import Tracker
import cv2
import torch


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tracker = Tracker(
        embedding_model_name='google/vit-base-patch16-224-in21k',
        yolo_model='yolov8m.pt',
        num_tables=10,
        max_add_images=10,
        multiprocessing=False,
        device=device,
        verbose=True
    )
    # Average time: 0.02482800139594324 Max time: 1.383957862854004
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        results = tracker.track(frame)
        results.plot()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.stop()



if __name__ == "__main__":
    main()