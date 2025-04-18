const User = require("../models/userModel");
const ResultMap = require("../models/resultMapModel");
const {generateJwtToken} = require("../services/auth");
const {join, dirname} = require("path");
const {readFileSync, createReadStream} = require("fs");
const axios = require("axios");
const fs = require("fs");
const { uuid } = require('uuidv4');
const FormData = require('form-data');

const FLASK_SERVER_ENDPOINT = "http://127.0.0.1:5001/predict";

async function handleSignUp(req, res) {
    if (!req.body.name || !req.body.email || !req.body.password)
        return res.status(400).json({
            error: "Please fill out all the fields!"
        });

    const existingUser = await User.findOne({email: req.body.email});

    if (existingUser)
        return res.status(400).json({
            error: "User with this email id already exists!"
        })

    const newUser = await User.create({
        name: req.body.name,
        email: req.body.email,
        password: req.body.password
    });

    const token = generateJwtToken(newUser);

    if (token) {
        res.cookie('token', token, {
            path: "/",
            expires: new Date(Date.now() + (24 * 7) * 3600000) // expires in 7 days
        })
        return res.status(201).json({
            'msg': "Success"
        })
    }

    return res.status(500).json({
        error: "Something went wrong"
    });
}

async function handleLogin(req, res) {
    if (!req.body.email || !req.body.password)
        return res.status(400).json({
            error: "Please fill out all the fields!"
        });

    const existingUser = await User.findOne({email: req.body.email});

    if (existingUser) {
        if (existingUser.password === req.body.password) {
            const token = generateJwtToken(existingUser);

            if (token) {
                res.cookie('token', token, {
                    path: "/",
                    expires: new Date(Date.now() + (24 * 7) * 3600000) // expires in 7 days
                })
                return res.status(201).json({
                    'msg': "Success"
                })
            }

            return res.status(500).json({
                error: "Something went wrong"
            });
        }
    }

    return res.status(400).json({
        error: "Invalid email or password!"
    });

}

async function handleAuthenticate(req, res) {
    const user = await User.findById(req.userId);

    return res.status(200).json({
        msg: "Authenticated",
        _id: user._id,
        name: user.name,
        email: user.email
    });

}

async function handleTakeTest(req, res) {
    try {
        console.log("Received take test request");
        
        // Check if file is uploaded
        if (!req.file) {
            console.log("No file in request");
            return res.status(400).json({ error: 'No file uploaded' });
        }

        console.log(`Processing file: ${req.file.filename}`);
        console.log(`File path: ${req.file.path}`);
        console.log(`File size: ${req.file.size} bytes`);
        console.log(`File mimetype: ${req.file.mimetype}`);

        // Create form data
        const formData = new FormData();
        formData.append('file', fs.createReadStream(req.file.path));

        console.log("Sending request to Flask server at:", FLASK_SERVER_ENDPOINT);
        
        try {
            // Add timeout and retry configuration
            const response = await axios.post(FLASK_SERVER_ENDPOINT, formData, {
                headers: {
                    ...formData.getHeaders(),
                    'Accept': 'application/json',
                    'Origin': 'http://localhost:8081'
                },
                maxContentLength: Infinity,
                maxBodyLength: Infinity,
                timeout: 30000, // 30 seconds timeout
                validateStatus: function (status) {
                    return status >= 200 && status < 500; // Accept all status codes less than 500
                }
            });

            console.log("Received response from Flask server");
            console.log("Response status:", response.status);
            console.log("Response data:", response.data);

            // Check if prediction was successful
            if (response.data.success) {
                // Read the image file and save it to the user's document
                const imageBuffer = fs.readFileSync(req.file.path);
                
                // Update the user document with the image and prediction results
                await User.findByIdAndUpdate(req.userId, {
                    image: imageBuffer,
                    imageType: req.file.mimetype,
                    resultPredictedClass: response.data.prediction,
                    resultPredictedProb: response.data.probability
                });

                // Clean up the uploaded file
                fs.unlink(req.file.path, (err) => {
                    if (err) console.error('Error deleting file:', err);
                });

                return res.json({
                    success: true,
                    prediction: response.data.prediction,
                    probability: response.data.probability,
                    description: response.data.description
                });
            } else {
                // Clean up the uploaded file
                fs.unlink(req.file.path, (err) => {
                    if (err) console.error('Error deleting file:', err);
                });
                
                return res.status(500).json({ 
                    error: 'Prediction failed',
                    details: response.data.error || 'Unknown error'
                });
            }
        } catch (error) {
            // Clean up the uploaded file
            fs.unlink(req.file.path, (err) => {
                if (err) console.error('Error deleting file:', err);
            });

            if (error.code === 'ECONNREFUSED') {
                console.error('Connection refused. Is the Flask server running?');
                return res.status(503).json({
                    error: 'Connection to prediction server failed',
                    details: 'The prediction server is not responding. Please try again later.'
                });
            }

            console.error('Error details:', error.message);
            return res.status(500).json({
                error: 'Error processing image',
                details: error.message
            });
        }
    } catch (error) {
        console.error('Error in handleTakeTest:', error);
        return res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
}

async function handleTestResults(req, res) {
    const doc = await User.findById(req.userId);

    return res.status(200).json({
        image: doc.image && doc.image.toString('base64'),
        imageType: doc.imageType,
        prediction: doc.resultPredictedClass,
        predictionConfidence: doc.resultPredictedProb
    })
}

async function handleSharedTestResults(req, res) {
    const key = req.params.key;

    const result = await ResultMap.findOne({key});

    if (result) {

        const doc = await User.findById(result.userId);

        return res.status(200).json({
            image: doc.image && doc.image.toString('base64'),
            imageType: doc.imageType,
            prediction: doc.resultPredictedClass,
            predictionConfidence: doc.resultPredictedProb
        })
    }

    return res.status(400).json({
        error: "Invalid Key"
    });
}

async function handleShareResults(req, res) {
    const map = await ResultMap.create({
        key: uuid(),
        userId: req.userId
    });

    return res.status(200).json({
        path: `/${map.key}`
    })

}

function handleLogout(req, res) {
    res.clearCookie('token');

    return res.status(200)
        .json({
            "msg": "Success"
        });
}

module.exports = {
    handleSignUp,
    handleLogin,
    handleAuthenticate,
    handleTakeTest,
    handleLogout,
    handleTestResults,
    handleSharedTestResults,
    handleShareResults
};