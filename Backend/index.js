const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const path = require('path');
const multer = require('multer');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const cookieParser = require('cookie-parser');
const fs = require('fs');
const FormData = require('form-data');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 8081;

// Middleware
app.use(cors({
    origin: ['http://localhost:5173', 'http://localhost:5174'],
    credentials: true
}));
app.use(express.json());
app.use(cookieParser());

// MongoDB Connection
const connectDB = async (retries = 5, interval = 5000) => {
    for (let i = 0; i < retries; i++) {
        try {
            console.log(`Attempting to connect to MongoDB (attempt ${i + 1}/${retries})...`);
            console.log('Connection URI:', process.env.MONGODB_URI);
            
            await mongoose.connect(process.env.MONGODB_URI, {
                serverSelectionTimeoutMS: 5000,
                socketTimeoutMS: 45000,
                family: 4 // Force IPv4
            });
            
            console.log('MongoDB connected successfully');
            
            // Test the connection
            const db = mongoose.connection;
            db.on('error', console.error.bind(console, 'MongoDB connection error:'));
            db.once('open', function() {
                console.log('MongoDB connection is open');
            });
            return;
        } catch (error) {
            console.error(`MongoDB connection attempt ${i + 1} failed:`, error);
            if (i < retries - 1) {
                console.log(`Retrying in ${interval/1000} seconds...`);
                await new Promise(resolve => setTimeout(resolve, interval));
            } else {
                console.error('All MongoDB connection attempts failed');
                console.error('Error details:', {
                    name: error.name,
                    message: error.message,
                    code: error.code,
                    codeName: error.codeName
                });
            }
        }
    }
};

// Connect to MongoDB
connectDB();

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/');
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter: (req, file, cb) => {
        // Accept only image files
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed!'), false);
        }
    }
});

// Import User model
const User = require('./models/userModel');

// Routes
app.post('/login', async (req, res) => {
    try {
        const { email, password } = req.body;
        console.log('Login attempt for:', email);

        const user = await User.findOne({ email });
        if (!user) {
            return res.status(401).json({ error: 'Invalid email or password' });
        }

        const isPasswordValid = await bcrypt.compare(password, user.password);
        if (!isPasswordValid) {
            return res.status(401).json({ error: 'Invalid email or password' });
        }

        // Create JWT token
        const token = jwt.sign(
            { userId: user._id, email: user.email },
            process.env.JWT_SECRET,
            { expiresIn: '24h' }
        );

        // Set cookie
        res.cookie('token', token, {
            httpOnly: true,
            secure: false, // Set to true in production
            sameSite: 'lax',
            maxAge: 24 * 60 * 60 * 1000 // 24 hours
        });

        res.json({
            success: true,
            user: {
                name: user.name,
                email: user.email
            }
        });
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.post('/signup', async (req, res) => {
    try {
        const { name, email, password } = req.body;
        console.log('Signup attempt for:', email);

        // Check if user already exists
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ error: 'User already exists' });
        }

        // Hash password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Create new user
        const newUser = await User.create({
            name,
            email,
            password: hashedPassword
        });

        // Create JWT token
        const token = jwt.sign(
            { userId: newUser._id, email: newUser.email },
            process.env.JWT_SECRET,
            { expiresIn: '24h' }
        );

        // Set cookie
        res.cookie('token', token, {
            httpOnly: true,
            secure: false, // Set to true in production
            sameSite: 'lax',
            maxAge: 24 * 60 * 60 * 1000 // 24 hours
        });

        res.json({
            success: true,
            user: {
                name: newUser.name,
                email: newUser.email
            }
        });
    } catch (error) {
        console.error('Signup error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.get('/authenticate', async (req, res) => {
    try {
        const token = req.cookies.token;
        if (!token) {
            return res.status(401).json({ error: 'No token provided' });
        }

        // Verify token
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        
        // Fetch user data from database
        const user = await User.findById(decoded.userId);
        if (!user) {
            return res.status(401).json({ error: 'User not found' });
        }

        res.json({
            authenticated: true,
            user: {
                name: user.name,
                email: user.email
            }
        });
    } catch (error) {
        console.error('Authentication error:', error);
        res.status(401).json({ error: 'Invalid token' });
    }
});

app.post('/logout', (req, res) => {
    res.clearCookie('token');
    res.json({ success: true, message: 'Logged out successfully' });
});

// Prediction route
app.post('/predict', upload.single('file'), async (req, res) => {
    try {
        console.log('Received prediction request');
        console.log('File:', req.file);
        
        if (!req.file) {
            return res.status(400).json({ error: 'No image uploaded' });
        }

        // Read the file
        const fileBuffer = fs.readFileSync(req.file.path);
        
        // Forward the request to Flask backend
        const flaskResponse = await fetch('http://127.0.0.1:5001/predict', {
            method: 'POST',
            body: fileBuffer,
            headers: {
                'Content-Type': req.file.mimetype,
                'Content-Disposition': `attachment; filename="${req.file.originalname}"`
            }
        });

        if (!flaskResponse.ok) {
            const errorText = await flaskResponse.text();
            console.error('Flask backend error:', errorText);
            throw new Error(`Flask backend responded with status: ${flaskResponse.status}`);
        }

        const prediction = await flaskResponse.json();
        
        // Clean up the uploaded file
        fs.unlinkSync(req.file.path);
        
        res.json(prediction);
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: 'Error processing prediction' });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`CORS enabled for: http://localhost:5173, http://localhost:5174`);
});