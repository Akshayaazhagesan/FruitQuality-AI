from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Load the saved model
model = load_model("Fruit_Quality_Classifier.h5")

# 2. Load your test image
img = image.load_img("apple.jpg", target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# 3. Predict
prediction = model.predict(img_array)
confidence = prediction[0][0]  # Get the confidence score

# 4. Show results in a friendly way
print("\n🍎 Fruit Quality Checker 🍏")
print(f"Confidence Score: {confidence:.4f}")

if confidence < 0.5:
    fresh_percent = (1 - confidence) * 100
    print(f"❌Prediction: ROTTEN ({fresh_percent:.1f}% sure)")
else:
    rotten_percent = confidence * 100
    print(f"✅ Prediction: FRESH ({rotten_percent:.1f}% sure)")

# 5. Extra explanation
if confidence > 0.4 and confidence < 0.6:
    print("\n⚠️ Note: The model isn't very sure about this one.")
elif confidence > 0.7 or confidence < 0.3:
    print("\n💡 Note: The model is quite confident about this prediction!")