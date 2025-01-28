const path = require('path');
const { app } = require('electron');

// Point to the electron.js file
const electronPath = app.isPackaged
  ? path.join(__dirname, 'electron.js')
  : path.join(__dirname, 'electron.js');

// Load the main electron process
require(electronPath);