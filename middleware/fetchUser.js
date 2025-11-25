const jwt = require('jsonwebtoken');
require('dotenv').config();

const fetchUser = (req, res, next) => {
    try {
        let token = null;

        const authHeader = req.headers.authorization;

        if (authHeader) {
            // Accept both "Bearer <token>" or raw token directly
            token = authHeader.startsWith('Bearer ') ? authHeader.split(' ')[1] : authHeader;
        } else if (req.header('auth-token')) {
            token = req.header('auth-token');
        }

        if (!token) {
            return res.status(401).json({
                success: false,
                message: "Access denied. No token provided.",
            });
        }

        const decoded = jwt.verify(token, process.env.JWT_SECRET);

        if (!decoded?.id || !decoded?.email) {
            return res.status(401).json({
                success: false,
                message: "Invalid token payload.",
            });
        }

        req.user = {
            id: decoded.id,
            email: decoded.email,
        };

        next();
    } catch (error) {
        console.error("JWT verification failed:", error);
        return res.status(401).json({
            success: false,
            message: "Invalid or expired token.",
            error: error.message,
        });
    }
};

module.exports = fetchUser;
