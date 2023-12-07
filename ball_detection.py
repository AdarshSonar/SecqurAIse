import cv2
import numpy as np

def detect_ball_color(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        
        return center, radius
    
    return None, None

def main():
    cap = cv2.VideoCapture('D:\SecqurAIse\AI Assignment video.mp4')  # video file
    
    color_ranges = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'blue': ([110, 50, 50], [130, 255, 255]),
        'green': ([40, 50, 50], [80, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'white': ([0, 0, 200], [180, 30, 255])  
    }
    
    # Dictionary to store entry and exit times for each ball
    ball_times = {color: {'entry': None, 'exit': None} for color in color_ranges.keys()}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for ball_color, (lower_color, upper_color) in color_ranges.items():
            center, radius = detect_ball_color(frame, np.array(lower_color), np.array(upper_color))
            
            if center is not None and radius is not None:
                # Draw circle and display color and time information
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.putText(frame, f'{ball_color.capitalize()} Ball', (center[0] - 50, center[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f'Time: {cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:.2f}s', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Check for entry time
                if ball_times[ball_color]['entry'] is None:
                    ball_times[ball_color]['entry'] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                
            else:
                # Check for exit time
                if ball_times[ball_color]['entry'] is not None and ball_times[ball_color]['exit'] is None:
                    ball_times[ball_color]['exit'] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    print(f'{ball_color.capitalize()} Ball - Entry: {ball_times[ball_color]["entry"]:.2f}s, Exit: {ball_times[ball_color]["exit"]:.2f}s')
                    
                    # Reset entry time for the next entry
                    ball_times[ball_color]['entry'] = None
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
