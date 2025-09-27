import tensorflow as tf
import cv2
import numpy as np
import os

class WastePredictor:
    def __init__(self, model_path, img_size=224):
        self.img_size = img_size
        self.class_names = ['Biodegradable', 'Non-Biodegradable']
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded!")
    
    def preprocess_image(self, frame):
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict_frame(self, frame):
        img = self.preprocess_image(frame)
        pred = self.model.predict(img, verbose=0)[0][0]  # verbose=0 avoids TF logs
        class_idx = 1 if pred > 0.5 else 0
        confidence = pred if class_idx == 1 else 1 - pred
        return self.class_names[class_idx], confidence
    
    def webcam_classify(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pred_class, conf = self.predict_frame(frame)
            label = f"{pred_class}: {conf*100:.2f}%"
            color = (0,255,0) if pred_class == 'Biodegradable' else (0,0,255)
            cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('Waste Classification', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    model_path = 'models/waste_classifier_mobilenetv2.h5'
    if not os.path.exists(model_path):
        print("❌ Model not found! Train the model first.")
        return
    predictor = WastePredictor(model_path)
    predictor.webcam_classify()

if __name__ == "__main__":
    main()
