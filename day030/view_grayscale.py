import cv2
import sys
import time

def main(camera_index=0):
    """
    Captures video from the specified camera, converts it to grayscale,
    and displays it in a window.
    """
    print(f"Attempting to open camera index: {camera_index}")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        print("Available cameras might be different. Try changing the index.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    window_name = f'Grayscale Camera View (Index: {camera_index}) - Press \'q\' to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Make window resizable

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the grayscale frame
        cv2.imshow(window_name, gray_frame)

        frame_count += 1

        # Wait for 1ms and check if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed, exiting...")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({fps:.2f} FPS).")


    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

if __name__ == "__main__":
    cam_idx = 0
    if len(sys.argv) > 1:
        try:
            cam_idx = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid camera index '{sys.argv[1]}'. Using default index 0.")

    main(cam_idx)
