const jwt = require('jsonwebtoken');
const env = require('dotenv');
env.config();

const fetchUser = (req, res, next) => {
    // Get user from the jwt token and add id to req object
    const token = req.header('auth-token');

    if (!token) {
        res.status(401).send({ error: "Please authenticate using a valid token" });
    }
    try {
        const data = jwt.verify(token, process.env.JWT_SECRET);
        req.user = data;
        next();
    } catch (error) {
        console.log(error);
        res.status(401).send({ error: "Please authenticate using a valid token" });
    }
}

module.exports = fetchUser;