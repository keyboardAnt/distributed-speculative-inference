const express = require('express');
const path = require('path');
const app = express();
const port = 3000;

// Serve the HTML file
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'memory_viz.html'));
});

// Serve the JavaScript file
app.get('/MemoryViz.js', (req, res) => {
  res.sendFile(path.join(__dirname, 'MemoryViz.js'));
});

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});