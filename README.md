# **Flare Scan**
Welcome to the Early Forest Fire Prediction project! This repository houses the groundbreaking initiative aimed at leveraging IoT and AI technologies to safeguard our forests from devastating wildfires.

### Abstract
The Early Forest Fire Prediction project is a comprehensive system designed to predict and confirm forest fires in their early stages, enabling swift response from forest management authorities. Utilizing a combination of IoT sensors, machine learning algorithms, and real-time communication modules, our solution aims to revolutionize wildfire prevention and forest preservation efforts.

### Problems we faced:
- From 2001 to 2022, India lost 3.59 lakh hectares of tree cover from fires and 2.15 million hectares from all other drivers of loss.
- The annual loss from forest fires in India is estimated to be around 440 crore rupees (about $107 million). This estimate does not include biodiversity loss, nutrient loss, soil moisture loss, and other intangible benefits.
- According to the forest inventory records, 54.40% of forests in India are exposed to occasional fires, 7.49% to moderately frequent fires, and 2.40% to high incidence levels.

### Previous research and Existing solution:
- **Satellite-Based Monitoring Systems**: Utilizing satellite imagery systems like MODIS and AVHRR for forest fire detection, but limited by low resolution and long scan periods.
- **Camera-Based Systems**: Implementing multi-sensor wireless networks with IP cameras for fire detection, requiring line-of-sight communication and elevated camera placement.
- **LIDAR Technology**: Employing Light Detection and Ranging (LIDAR) for fire detection by measuring laser light backscattered by smoke particles, yet facing limitations such as high false alert rates due to climatic conditions.
- **Wireless Sensor Network (WSN)**: Developing systems based on WSN for forest fire identification, predominantly relying on temperature and humidity parameters, but overlooking other influential factors like wind.

## Workflow
![Workflow](C:/Users/Sarthak Aggarwal/Downloads/Telegram Desktop/photo_2024-03-31_09-13-43.jpg)

## Implementation
### Hardware Components:
- Smart sensor box equipped with temperature, humidity, rainfall, and wind speed sensors.
- GSM module for real-time communication of fire location.
- Camera for capturing images of potential fire outbreaks.
### Software Components:
- Integration with Firebase for real-time data transmission.
- Support Vector Machine (SVM)-based ML model for fire prediction.
- OpenCV for image processing.
- YOLOv5 for fire confirmation.
### Workflow:
- IoT sensors gather environmental data and transmit it to Firebase via WiFi.
- ML model analyzes the data to predict the likelihood of a fire outbreak.
- Upon detection of a potential threat, the system activates the camera to capture images.
- Images are processed using YOLOv5 for confirmation of fire presence.
- Instant alerts are dispatched to the forest department, along with the precise fire location relayed through the GSM module.

## Working
Introducing "Early Forest Fire Prediction" – a groundbreaking project leveraging IoT and AI technologies to safeguard our forests. Our smart sensor box, equipped with temperature, humidity, rainfall, and wind speed sensors, transmits real-time data via WiFi to Firebase. The data is seamlessly integrated into our SVM-based ML model, predicting the likelihood of a fire outbreak. Upon detecting a potential threat, our system activates a camera capturing images, processed by YOLOv5 through HTTP, confirming the presence of fire.
In the event of a confirmed fire, instant alerts are dispatched to the forest department, ensuring swift response. Simultaneously, our GSM module relays the precise fire location, expediting the rescue efforts. This comprehensive solution not only facilitates early detection but also employs cutting-edge technologies like OpenCV and YOLOv5 for rapid and accurate fire confirmation. A fusion of environmental monitoring, machine learning, and real-time communication, our system is a vital step towards proactive forest preservation. Join us in revolutionizing wildfire prevention for a greener future.

### Android UI
![UI](https://github.com/your_username/your_repository/raw/main/acv.jpg)

### Why our Project:
- **Scalability:** Seamlessly expandable, our modular forest fire detection system adapts to diverse environments, extending coverage and staying ahead of evolving threats for agile, responsive protection.
- **Feasibility:** Our cost-effective solution integrates accessible technologies, ensuring practicality and affordability for stakeholders. Simple, efficient, and accessible—empowering communities to safeguard forests effectively.
- **Long-Term Vision:** Beyond detection, our project fosters sustainable forest management, empowering stakeholders to shape policy and enhance ecosystem resilience for a brighter, greener future.

### Challenges we faced:
- **Data Accuracy:** Ensuring accurate and reliable data from IoT sensors in diverse environmental conditions.
- **Cost and Accessibility:** Procuring and deploying a sufficient number of sensors across vast forested areas can be costly and logistically challenging. Ensuring availability of sensors within budget constraints while maintaining coverage over critical areas poses a significant challenge.
- **Maintenance and Reliability:** Remote forest environments present harsh conditions that can affect the longevity and reliability of sensors. Ensuring continuous availability and functionality of sensors amidst factors like extreme weather, wildlife interference, and physical damage requires robust maintenance strategies and resilient sensor designs.
- **Real-time Communication:** Overcoming latency issues in transmitting data and alerts for swift response.
- **Integration Complexity:** Integrating multiple hardware and software components seamlessly for a cohesive system.

### Future Enhancements
- Implementing advanced AI algorithms for more accurate fire detection.
- Expanding sensor capabilities for a broader range of environmental data collection.
- Enhancing real-time communication protocols for faster response times.
- Collaborating with forest management authorities to integrate additional features based on their requirements.

# THANK YOU!!
