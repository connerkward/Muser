// Test script to verify P2P WebRTC functionality
const { spawn } = require('child_process');
const path = require('path');

console.log('Testing P2P WebRTC functionality...\n');
console.log('This will open 2 Electron windows that should connect via WebRTC');
console.log('They should share the same room and sync drawing changes\n');

// Build first
console.log('Building the app...');
const build = spawn('npm', ['run', 'build'], {
  cwd: __dirname,
  stdio: 'inherit',
  shell: true
});

build.on('close', (code) => {
  if (code !== 0) {
    console.error('Build failed');
    process.exit(1);
  }
  
  console.log('\nBuild complete. Starting 2 instances...\n');
  
  // Start first instance
  setTimeout(() => {
    console.log('Starting instance 1...');
    const instance1 = spawn('npm', ['start'], {
      cwd: __dirname,
      stdio: 'inherit',
      shell: true
    });
  }, 1000);
  
  // Start second instance
  setTimeout(() => {
    console.log('Starting instance 2...');
    const instance2 = spawn('npm', ['start'], {
      cwd: __dirname,
      stdio: 'inherit',
      shell: true
    });
  }, 3000);
  
  console.log('\nTest Instructions:');
  console.log('1. Both windows should open');
  console.log('2. Use the same room ID in both windows');
  console.log('3. Draw in one window - it should appear in the other');
  console.log('4. Check the connection status shows "Connected" and "1 peer"');
  console.log('\nPress Ctrl+C to stop all instances\n');
});


