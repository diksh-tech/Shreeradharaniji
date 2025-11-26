import React, { useState } from 'react';
import { MapPin, Clock, Calendar, Plane } from 'lucide-react';

const RouteSelection = ({ routes, message, onRouteSelect }) => {
  const [selectedRoute, setSelectedRoute] = useState(null);

  const formatTime = (timeString) => {
    if (!timeString) return 'N/A';
    try {
      const date = new Date(timeString);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return timeString;
    }
  };

  const handleSelect = (routeIndex) => {
    setSelectedRoute(routeIndex);
    onRouteSelect(routeIndex);
  };

  return (
    <div className="flex justify-start mb-4">
      <div className="assistant-message max-w-2xl">
        <div className="flex items-center space-x-2 mb-3">
          <MapPin className="h-5 w-5 text-blue-500" />
          <span className="font-medium text-gray-800">Multiple Routes Found</span>
        </div>
        
        <p className="text-gray-600 mb-4">{message}</p>

        <div className="space-y-3">
          {routes.map((route, index) => (
            <div
              key={index}
              onClick={() => handleSelect(index)}
              className={`p-4 border-2 rounded-xl cursor-pointer transition-all duration-200 ${
                selectedRoute === index
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300 hover:bg-blue-25'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-2">
                    <MapPin className="h-4 w-4 text-green-600" />
                    <span className="font-semibold">{route.startStation}</span>
                  </div>
                  <Plane className="h-4 w-4 text-gray-400" />
                  <div className="flex items-center space-x-2">
                    <MapPin className="h-4 w-4 text-red-600" />
                    <span className="font-semibold">{route.endStation}</span>
                  </div>
                </div>
                <div className={`px-2 py-1 rounded text-xs font-medium ${
                  selectedRoute === index
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-600'
                }`}>
                  Option {index + 1}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                <div className="flex items-center space-x-2">
                  <Clock className="h-4 w-4" />
                  <span>Depart: {formatTime(route.scheduledStartTime)}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Clock className="h-4 w-4" />
                  <span>Arrive: {formatTime(route.scheduledEndTime)}</span>
                </div>
              </div>

              {route.carrier && route.flightNumber && (
                <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                  <span>Carrier: {route.carrier}</span>
                  <span>Flight: {route.flightNumber}</span>
                  {route.dateOfOrigin && (
                    <div className="flex items-center space-x-1">
                      <Calendar className="h-3 w-3" />
                      <span>{route.dateOfOrigin}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <p className="text-sm text-amber-800 text-center">
            Please select your desired route to continue
          </p>
        </div>
      </div>
    </div>
  );
};

export default RouteSelection;
