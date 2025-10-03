import React from 'react';

const Navbar = ({ systemStatus, onTestPrediction, onTestAlert }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'connected':
      case 'healthy':
        return 'bg-green-500';
      case 'connecting':
        return 'bg-yellow-500 animate-pulse';
      case 'disconnected':
        return 'bg-red-500';
      case 'error':
        return 'bg-red-600';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <nav className="bg-mine-800 text-white shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Title */}
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <h1 className="text-xl font-bold">RockWatch AI</h1>
              <p className="text-xs text-mine-300">Rockfall Prediction System</p>
            </div>
          </div>

          {/* System Status */}
          <div className="flex items-center space-x-6">
            <div className="flex items-center">
              <span className={`inline-block w-3 h-3 rounded-full mr-2 ${getStatusColor(systemStatus.status)}`}></span>
              <span className="text-sm">
                System: {systemStatus.status}
              </span>
            </div>

            <div className="text-sm">
              <span className="text-mine-300">Models:</span> {systemStatus.modelsLoaded.length}
            </div>

            <div className="text-sm">
              <span className="text-mine-300">Uptime:</span> {systemStatus.uptime}
            </div>

            <div className="text-sm">
              <span className="text-mine-300">Clients:</span> {systemStatus.connectedClients}
            </div>

            {/* Test Controls */}
            <div className="flex items-center space-x-2">
              <button
                onClick={onTestPrediction}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
              >
                Test Prediction
              </button>
              
              <div className="relative group">
                <button className="px-3 py-1 bg-orange-600 hover:bg-orange-700 rounded text-sm font-medium transition-colors">
                  Test Alert ▼
                </button>
                <div className="absolute right-0 mt-2 w-32 bg-white rounded-md shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                  <button
                    onClick={() => onTestAlert('warning')}
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-yellow-50 hover:text-yellow-800"
                  >
                    Warning
                  </button>
                  <button
                    onClick={() => onTestAlert('critical')}
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-red-50 hover:text-red-800"
                  >
                    Critical
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;