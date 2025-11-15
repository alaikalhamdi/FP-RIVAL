import argparse
import cv2
import cv2.aruco as aruco

def parse_args():
    p = argparse.ArgumentParser(description="Simple ArUco detector (webcam, video, or phone stream)")
    p.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--video", type=str, help="Path to input video file (optional)")
    p.add_argument("--phone-url", type=str, default="http://192.168.1.100:8080/video", 
                   help="URL to phone camera stream (e.g. http://IP:PORT/video, rtsp://, etc.)")
    p.add_argument("--dict", type=str, default="6x6_250", help="Aruco dictionary name (e.g. 4x4_50, 6x6_250)")
    p.add_argument("--no-display", action="store_true", help="Do not open a display window (headless)")
    return p.parse_args()

def main(argv=None):
    args = parse_args() if argv is None else argv
    # --- Select your ArUco dictionary ---
    dict_map = {
        "4x4_50": aruco.DICT_4X4_50,
        "4x4_100": aruco.DICT_4X4_100,
        "4x4_250": aruco.DICT_4X4_250,
        "5x5_100": aruco.DICT_5X5_100,
        "6x6_250": aruco.DICT_6X6_250,
    }
    key = (args.dict or "6x6_250").lower().replace("-", "_")
    aruco_name = dict_map.get(key, aruco.DICT_6X6_250)
    aruco_dict = aruco.getPredefinedDictionary(aruco_name)
    # create detector parameters (DetectorParameters or DetectorParameters_create depending on cv2)
    try:
        if hasattr(aruco, "DetectorParameters_create"):
            parameters = aruco.DetectorParameters_create()
        else:
            parameters = aruco.DetectorParameters()
    except Exception:
        parameters = None
    # --- Create the detector object ---
    try:
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        use_detector = True
    except Exception:
        detector = None
        use_detector = False
    # --- Open the video source ---
    # Priority: phone-url > video > camera
    if getattr(args, "phone_url", None):
        print(f"Opening phone stream: {args.phone_url}")
        cap = cv2.VideoCapture(args.phone_url)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    window_name = "ArUco Marker Detection"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # --- Convert to grayscale ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # --- Detect markers ---
        try:
            if use_detector and detector is not None:
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        except Exception:
            corners, ids, rejected = None, None, None
        # --- Draw markers ---
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            # Annotate each marker ID
            for i in range(len(ids)):
                c = corners[i][0]
                cx = int(c[:, 0].mean())
                cy = int(c[:, 1].mean())
                cv2.putText(frame, f"ID: {ids[i][0]}", (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # --- Show result ---
        if not args.no_display:
            cv2.imshow(window_name, frame)
        # Quit with 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
