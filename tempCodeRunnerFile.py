# Step 4: Apply region of interest
        masked_edges = self.region_of_interest(edges)
        print(f"process_frame: Applied ROI to edges.")
        cv2.imshow('Masked Edges', masked_edges)