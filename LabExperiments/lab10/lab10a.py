import cv2
import numpy as np

def compute_gradients(frame_prev, frame_curr, frame_next):

    I_prev = frame_prev.astype(np.float32) / 255.0
    I_curr = frame_curr.astype(np.float32) / 255.0
    I_next = frame_next.astype(np.float32) / 255.0
    
    # Second-order accurate central temporal derivative
    I_t = 0.5 * (I_next - I_prev)
    
    # Spatial derivatives averaged across all three frames to minimize noise
    Ix_prev, Iy_prev = np.gradient(I_prev)
    Ix_curr, Iy_curr = np.gradient(I_curr)
    Ix_next, Iy_next = np.gradient(I_next)
    
    I_x = (Ix_prev + Ix_curr + Ix_next) / 3.0
    I_y = (Iy_prev + Iy_curr + Iy_next) / 3.0
    
    return I_x, I_y, I_t

def lucas_kanade(prev_gray, curr_gray, next_gray, points, window_size=15):
   
    # Compute 3-frame central gradients centered at time t (curr_gray)
    I_x, I_y, I_t = compute_gradients(prev_gray, curr_gray, next_gray)
    
    half_w = window_size // 2
    h, w = curr_gray.shape
    
    valid_prev = []
    valid_next = []
    speeds = []
    
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        
        # Ensure window stays within image bounds
        if x - half_w < 0 or x + half_w >= w or y - half_w < 0 or y + half_w >= h:
            continue
            
        # Extract local spatial and temporal windows around the point
        patch_x = I_x[y - half_w : y + half_w + 1, x - half_w : x + half_w + 1].flatten()
        patch_y = I_y[y - half_w : y + half_w + 1, x - half_w : x + half_w + 1].flatten()
        patch_t = I_t[y - half_w : y + half_w + 1, x - half_w : x + half_w + 1].flatten()
        
        # Construct matrix A (N x 2) and vector b (N x 1)
        A = np.vstack((patch_x, patch_y)).T
        b = -patch_t.reshape(-1, 1)
        
        # Compute Structure Tensor M = A^T * A
        M = np.dot(A.T, A)
        
        # Eigenvalue thresholding to reject flat areas and single edges (Aperture Problem)
        eigenvalues = np.linalg.eigvals(M)
        if np.min(eigenvalues) < 0.01:
            continue
            
        try:
            # Solve normal equations: v = (A^T A)^-1 A^T b
            v = np.dot(np.dot(np.linalg.inv(M), A.T), b)
            u_vel, v_vel = v[0, 0], v[1, 0]
            
            speed = np.sqrt(u_vel**2 + v_vel**2)
            if speed > 20.0:  
                continue
                
            next_x = x + u_vel
            next_y = y + v_vel
            
            valid_prev.append([x, y])
            valid_next.append([next_x, next_y])
            speeds.append(speed)
            
        except np.linalg.LinAlgError:
            continue
            
    return (
        np.array(valid_prev, dtype=np.float32), 
        np.array(valid_next, dtype=np.float32), 
        np.array(speeds, dtype=np.float32)
    )


def run_3frame_pipeline(video_path):
    cap = cv2.VideoCapture(video_path)
    
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    ret3, frame3 = cap.read()
    
    if not (ret1 and ret2 and ret3):
        print("Video too short for 3-frame central difference.")
        return
        
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    
    h, w = curr_gray.shape
    step = 20
    grid_y, grid_x = np.mgrid[step:h-step:step, step:w-step:step]
    initial_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T.astype(np.float32)
    
    while cap.isOpened():
        # Compute optical flow centered at curr_gray using prev_gray and next_gray
        prev_pts, next_pts, speeds = lucas_kanade(prev_gray, curr_gray, next_gray, initial_points)

        avg_speed=np.mean(speeds) if len(speeds)>0 else 0.0
        # Render results on curr_gray frame...
        display_frame = frame2.copy()
        for pt_p, pt_n, spd in zip(prev_pts, next_pts, speeds):
            p1 = (int(pt_p[0]), int(pt_p[1]))
            p2 = (int(pt_n[0]), int(pt_n[1]))
            cv2.arrowedLine(display_frame, p1, p2, (0, 0, 255), 2, tipLength=0.3)


        hud_text = f"Active Vectors: {len(prev_pts)} | Avg Speed: {avg_speed:.2f} px/frame"
        cv2.putText(display_frame, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)    
        cv2.imshow("3-Frame Lucas-Kanade", display_frame)

        
        
        # Shift the rolling window forward by one frame
        prev_gray = curr_gray
        curr_gray = next_gray
        ret, new_frame = cap.read()
        if not ret:
            break
        frame2 = frame3
        frame3 = new_frame
        next_gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        
        if len(next_pts) > 0:
            initial_points = next_pts
        else:
            initial_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T.astype(np.float32)
            
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


video_path="test_moving_box.mp4"
run_3frame_pipeline(video_path)