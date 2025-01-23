Overview
This project aims to calculate the volume of a circular crop field using satellite imagery and GNSS (Global Navigation Satellite System) measurements. The process involves detecting the crop circle's edge, converting pixel coordinates to geospatial coordinates, and calculating the volume using mathematical integration techniques.

Components:
  *Satellite Image Retrieval: Obtain high-resolution satellite imagery centered on a specified location.

  *Edge Detection: Detect the crop circle's edge from the satellite image.

  *Coordinate Transformation: Convert pixel coordinates to geospatial coordinates.

  *Data Capture: Capture radius and height data around the circle's perimeter.

  *Interpolation: Interpolate the data for smooth transitions.

  *Volume Calculation: Calculate the volume using mathematical integration.

  *Visualization: Visualize the crop circle and the volume calculation results.

1. Satellite Image Retrieval
Description:
Obtain a satellite image centered on a specified latitude and longitude using the Google Maps API.

CODE:
###########################################################################
import requests
from io import BytesIO
from PIL import Image

def get_satellite_image(latitude, longitude, zoom=18, size=(640, 640)):
    api_key = 'YOUR_API_KEY'  # Replace with your API key
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size[0]}x{size[1]}&maptype=satellite&key={api_key}"

    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Example usage:
latitude = 40.7128   # Your center point latitude
longitude = -74.0060 # Your center point longitude

satellite_image = get_satellite_image(latitude, longitude)
satellite_image.show()

##########################################################################


Explanation:
    *get_satellite_image Function: This function takes the latitude and longitude of the center point, a zoom level, and image size to fetch a satellite image           using the Google Maps API.

    *API Key: Replace 'YOUR_API_KEY' with your actual Google Maps API key.

    *Example Usage: Retrieve and display a satellite image of New York City.

2. Edge Detection
Description:
Detect the crop circle's edge by analyzing pixel color changes on the satellite image.

Code:
#########################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_crop_circle(satellite_image):
    image_cv = cv2.cvtColor(np.array(satellite_image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

    circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image_cv, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image_cv, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            center = (x, y)
            radius = r
        plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        return center, radius
    else:
        print("No crop circle detected")
        return None, None

# Detect the crop circle
center_pixel, radius_pixel = detect_crop_circle(satellite_image)
#########################################################################

Explanation:
  *detect_crop_circle Function: Converts the image to grayscale, applies Gaussian blur, and uses the Hough Circle Transform to detect circles.

  *Hough Circle Transform: Detects circular shapes in the image.

  *Visual Feedback: Draws detected circles and displays the image.

3. Coordinate Transformation
Description:
Convert the pixel-based center and radius to real-world latitude and longitude coordinates.

Code:

#########################################################################
def calculate_scale(latitude, zoom):
    equator_length = 40075016.686
    meters_per_pixel = equator_length * np.cos(np.deg2rad(latitude)) / (2 ** (zoom + 8))
    return meters_per_pixel

meters_per_pixel = calculate_scale(latitude, zoom=18)
actual_radius_meters = radius_pixel * meters_per_pixel
#########################################################################

Explanation:
  *calculate_scale Function: Computes the scale (meters per pixel) based on latitude and zoom level.

  *Actual Radius: Converts pixel radius to meters using the calculated scale.

4. Generating Radial Coordinates
Description:
Generate a set of points around the circle's edge to collect GNSS height data.

Code:
#########################################################################
def generate_circle_points(center_lat, center_lon, radius_meters, num_points=360):
    points = []
    angles = np.linspace(0, 2 * np.pi, num_points)
    for angle in angles:
        delta_lat = (radius_meters * np.cos(angle)) / 110574  # Approximate meters per degree latitude
        delta_lon = (radius_meters * np.sin(angle)) / (111320 * np.cos(np.deg2rad(center_lat)))  # Approximate meters per degree longitude
        lat = center_lat + delta_lat
        lon = center_lon + delta_lon
        points.append((lat, lon))
    return angles, points

angles_rad, circle_points = generate_circle_points(latitude, longitude, actual_radius_meters)
#########################################################################

Explanation:
    *generate_circle_points Function: Generates latitude and longitude coordinates around the circle's perimeter based on the radius.

    *Geospatial Transformation: Converts distances in meters to degrees of latitude and longitude.

5. Acquiring GNSS Height Data
Description:
Retrieve elevation data for each point around the circle using an elevation API.

Code:
#########################################################################
def get_elevation_data(points):
    elevations = []
    for lat, lon in points:
        elevation = fetch_elevation_from_api(lat, lon)  # Implement this function
        elevations.append(elevation)
    return elevations

def fetch_elevation_from_api(lat, lon):
    url = f'https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}'
    response = requests.get(url).json()
    elevation = response['results'][0]['elevation']
    return elevation

height_data = get_elevation_data(circle_points)
#########################################################################
Explanation:
    *get_elevation_data Function: Iterates through circle points and retrieves elevation data using an API.

    *Example API Call: Uses Open-Elevation API to fetch elevation data.
6. Interpolating Values
Description:
Interpolate the radius and height data for smooth transitions.

Code:
#########################################################################
from scipy.interpolate import interp1d

def interpolate_values(theta, values):
    interp_func = interp1d(theta, values, kind='cubic')
    fine_theta = np.linspace(0, 2 * np.pi, 3600)  # Higher resolution
    fine_values = interp_func(fine_theta)
    return fine_theta, fine_values

fine_theta, fine_radius = interpolate_values(np.degrees(angles_rad), [actual_radius_meters] * len(angles_rad))
_, fine_height = interpolate_values(np.degrees(angles_rad), height_data)
#########################################################################
Explanation:
*interpolate_values Function: Uses cubic interpolation to generate a higher resolution data set for radius and height.

*Fine Resolution: Provides smoother transitions between data points.

7. Volume Calculation
Description:
Calculate the volume using the interpolated radius and height data.

Code:
#########################################################################
def calculate_volume(fine_theta, fine_radius, fine_height):
    volume = 0
    for i in range(len(fine_theta) - 1):
        r1 = fine_radius[i]
        r2 = fine_radius[i + 1]
        h1 = fine_height[i]
        h2 = fine_height[i + 1]
        d_theta = np.deg2rad(fine_theta[i + 1] - fine_theta[i])
        volume += (1/3) * d_theta * (r1**2 + r1*r2 + r2**2) * (h1 + h2) / 2
    return volume

# Calculate the volume
volume = calculate_volume(fine_theta, fine_radius, fine_height)
print(f"Calculated Volume: {volume:.5f} cubic meters")
#########################################################################

Explanation:
*calculate_volume Function: Uses the method of disks to calculate the volume of the crop field.

*Integration: Computes the volume by summing the contributions of infinitesimal wedges.
(see mathematical explaination in concept.pdf)
