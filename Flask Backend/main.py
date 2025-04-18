import os
import numpy as np
from datetime import datetime, timedelta
import jwt
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from PIL import Image
import logging
import io

# Define class names
class_names = ['melanoma', 'nevus', 'basal cell carcinoma', 'actinic keratosis', 
               'squamous cell carcinoma', 'pigmented benign keratosis', 'seborrheic keratosis']

def create_model(input_shape=(224, 224, 3), num_classes=7):
    # Load the pre-trained MobileNetV2 model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(train_dir, validation_dir, epochs=50, batch_size=32):
    # Create data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Create and train the model
    model = create_model()
    
    # Add early stopping
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[early_stopping, checkpoint]
    )

    return model, history

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Configure CORS with all necessary settings
CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:8081", "http://localhost:5173", "http://localhost:5174"],
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
         "supports_credentials": True,
         "expose_headers": ["Set-Cookie"],
         "allow_credentials": True
     }},
     supports_credentials=True)

# Add basic request logging
@app.before_request
def log_request():
    print(f"\n{'='*50}")
    print(f"Received {request.method} request to {request.path}")
    print(f"Headers: {dict(request.headers)}")
    if request.method == 'POST':
        print(f"Form data: {request.form}")
        print(f"Files: {request.files}")
    print(f"{'='*50}\n")

# Add test route
@app.route('/test', methods=['GET'])
def test_route():
    print("Test route called")
    return jsonify({
        'status': 'success',
        'message': 'Flask server is running!',
        'endpoints': {
            'predict': '/predict',
            'signup': '/signup',
            'login': '/login'
        }
    })

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
print("Loading model...")
try:
    # Clear any existing TensorFlow session
    tf.keras.backend.clear_session()
    
    model_path = os.path.join(os.path.dirname(__file__), 'my_skin_disease_pred_model.h5')
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    print("Model loaded successfully!")
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
    print("Model summary:")
    model.summary()
    
    # Verify model architecture
    if not isinstance(model, tf.keras.Model):
        raise ValueError("Loaded model is not a Keras model")
        
    # Check if model has the expected number of output classes
    output_shape = model.output_shape
    if isinstance(output_shape, tuple):
        num_classes = output_shape[-1]
    else:
        num_classes = output_shape[0][-1]
        
    print(f"Number of output classes: {num_classes}")
    print(f"Expected number of classes: {len(class_names)}")
    
    if num_classes != len(class_names):
        raise ValueError(f"Model output classes ({num_classes}) does not match expected classes ({len(class_names)})")
        
except Exception as e:
    print(f"Error loading model: {str(e)}")
    import traceback
    print(traceback.format_exc())
    raise

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('token')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE id = ?', (data['user_id'],))
            current_user = c.fetchone()
            conn.close()
            if not current_user:
                return jsonify({'error': 'User not found'}), 401
        except:
            return jsonify({'error': 'Token is invalid'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not all([name, email, password]):
        return jsonify({'error': 'Missing required fields'}), 400

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        hashed_password = generate_password_hash(password)
        c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                 (name, email, hashed_password))
        conn.commit()
        return jsonify({'message': 'User created successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists'}), 400
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({'error': 'Missing email or password'}), 400

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()

    if not user or not check_password_hash(user[3], password):
        return jsonify({'error': 'Invalid email or password'}), 401

    token = jwt.encode({
        'user_id': user[0],
        'exp': datetime.utcnow() + timedelta(days=1)
    }, app.config['SECRET_KEY'])

    # Return user data in the response
    user_data = {
        'id': user[0],
        'name': user[1],
        'email': user[2]
    }

    response = jsonify({
        'message': 'Login successful',
        'user': user_data
    })
    
    # Set cookie with correct settings for cross-origin
    response.set_cookie(
        'token',
        token,
        httponly=True,
        secure=False,  # Set to False for development
        samesite='Lax',
        path='/',
        max_age=86400  # 24 hours
    )
    
    # Add CORS headers explicitly
    origin = request.headers.get('Origin')
    if origin:
        response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    
    return response

@app.route('/authenticate', methods=['GET'])
def authenticate():
    token = request.cookies.get('token')
    if not token:
        return jsonify({'error': 'Token is missing'}), 401
    try:
        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        user_id = data.get('user_id')
        
        # Get user data from the database
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'User not found'}), 401
            
        # Return user data in the response
        user_data = {
            'id': user[0],
            'name': user[1],
            'email': user[2]
        }
        
        response = jsonify({
            'authenticated': True,
            'user': user_data
        })
        
        # Add CORS headers explicitly
        origin = request.headers.get('Origin')
        if origin:
            response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        
        return response, 200
    except:
        return jsonify({'error': 'Token is invalid'}), 401

@app.route('/logout', methods=['POST'])
def logout():
    response = jsonify({'message': 'Logged out successfully'})
    response.delete_cookie('token')
    
    # Add CORS headers explicitly
    origin = request.headers.get('Origin')
    if origin:
        response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    
    return response

@app.route('/train', methods=['POST'])
@token_required
def train():
    try:
        if 'train_data' not in request.files or 'validation_data' not in request.files:
            return jsonify({'error': 'Please provide both training and validation data'}), 400

        train_data = request.files['train_data']
        validation_data = request.files['validation_data']

        # Create temporary directories for training
        train_dir = 'temp/train'
        validation_dir = 'temp/validation'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(validation_dir, exist_ok=True)

        # Save the uploaded data
        train_path = os.path.join(train_dir, train_data.filename)
        validation_path = os.path.join(validation_dir, validation_data.filename)
        train_data.save(train_path)
        validation_data.save(validation_path)

        # Train the model
        print("Starting model training...")
        model, history = train_model(train_dir, validation_dir)
        
        # Save the trained model
        model.save('my_skin_disease_pred_model.h5')
        
        # Clean up temporary files
        import shutil
        shutil.rmtree('temp')

        return jsonify({
            'message': 'Model trained successfully',
            'accuracy': history.history['accuracy'][-1],
            'val_accuracy': history.history['val_accuracy'][-1]
        })

    except Exception as e:
        print(f"Error in training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Error during training: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n" + "="*50)
        print("Received prediction request")
        print("Request headers:", dict(request.headers))
        
        if not request.data:
            print("No data received")
            return jsonify({'error': 'No data received'}), 400
            
        try:
            # Read the image from request data
            image = Image.open(io.BytesIO(request.data))
            print("Image opened successfully")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                print("Converting image to RGB")
                image = image.convert('RGB')
            
            # Preprocess the image - resize to 64x64 as expected by the model
            print("Resizing image to 64x64")
            image = image.resize((64, 64))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            print("Image array shape:", image_array.shape)
            
            # Make prediction
            print("Making prediction")
            prediction = model.predict(image_array)
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            
            # Map class index to label
            class_names = ['Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 
                         'Actinic keratosis', 'Benign keratosis', 'Dermatofibroma', 
                         'Vascular lesion']
            
            result = {
                'class': class_names[predicted_class],
                'confidence': confidence
            }
            
            print("Prediction successful")
            print("Result:", result)
            return jsonify(result)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

def get_class_description(class_name):
    descriptions = {
        'melanoma': 'A serious form of skin cancer that can be life-threatening if not treated early.',
        'nevus': 'A common type of mole that is usually harmless.',
        'basal cell carcinoma': 'The most common type of skin cancer, usually slow-growing and treatable.',
        'actinic keratosis': 'Precancerous skin growths caused by sun damage.',
        'squamous cell carcinoma': 'A type of skin cancer that can grow and spread more quickly than basal cell carcinoma.',
        'pigmented benign keratosis': 'Non-cancerous skin growths that are usually harmless.',
        'seborrheic keratosis': 'Common non-cancerous skin growths that appear with age.'
    }
    return descriptions.get(class_name, 'No description available.')

if __name__ == '__main__':
    # Enable debug mode
    app.debug = True
    
    # Run the app with specific host and port
    print("="*50)
    print("Starting Flask server...")
    print("Server will run on http://localhost:5001")
    print("Debug mode is enabled")
    print("CORS is configured for:")
    print("- http://localhost:8081")
    print("- http://localhost:5173")
    print("- http://localhost:5174")
    print("="*50)
    
    app.run(host='localhost', port=5001, debug=True)