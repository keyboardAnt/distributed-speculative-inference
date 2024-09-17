const express = require('express');
const path = require('path');
const fs = require('fs');
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

// Serve a file from an absolute path
app.get('/load-file', (req, res) => {
  const filePath = req.query.path;
  if (!filePath) {
    res.status(400).json({ error: 'Path query parameter is required' });
    return;
  }

  const absolutePath = path.resolve(filePath);
  console.log(`Loading file: ${absolutePath}`);

  fs.access(absolutePath, fs.constants.R_OK, (err) => {
    if (err) {
      console.error(`Error accessing file: ${err.message}`);
      res.status(404).json({ error: 'File not found or inaccessible' });
      return;
    }

    res.sendFile(absolutePath, (err) => {
      if (err) {
        console.error(`Error sending file: ${err.message}`);
        res.status(500).json({ error: 'Error sending file' });
      }
    });
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});