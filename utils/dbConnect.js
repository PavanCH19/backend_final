const mongoose = require('mongoose');

const dbConnect = async () => {

    if (!process.env.MONGO_URL) {
        throw new Error('MONGO_URI is not defined in environment variables');
    }

    const conn = await mongoose.connect(process.env.MONGO_URL)
        .then(() => console.log('MongoDB connected...'))
        .catch((error) => {
            console.error(`Error: ${error.message}`);
            process.exit(1);
        });

    return conn;
}

module.exports = dbConnect;