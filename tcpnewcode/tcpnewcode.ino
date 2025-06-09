#define BLYNK_PRINT Serial
#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
#include <Wire.h>
#include <MPU9250_asukiaaa.h>

// Blynk Legacy
char auth[] = "s6twEQqusbfwEcrQYdID2avZ00GTA_r8";
char ssid[] = "Adawiyah";
char pass[] = "12345678";

// Blynk server
IPAddress server_ip_blynk(152, 42, 239, 200);
int blynk_port = 8080;

// TCP server Python
IPAddress server_ip_tcp(172,20,10,10); // IP Python PC
int tcp_port = 3000;

WiFiClient tcpClient;
BlynkTimer timer;
MPU9250_asukiaaa mySensor;

// FSR pins
const int fsrPin1 = 32, fsrPin2 = 35, fsrPin3 = 33, fsrPin4 = 34;
float fsrForce1, fsrForce2, fsrForce3, fsrForce4;

// Moving average filter for FSR
const int numReadings = 5;
float fsrReadings1[numReadings], fsrReadings2[numReadings], fsrReadings3[numReadings], fsrReadings4[numReadings];
int readIndex = 0;
float fsrTotal1 = 0, fsrTotal2 = 0, fsrTotal3 = 0, fsrTotal4 = 0;

String gaitLabel = "Normal";
bool isStreaming = false; // Kawalan dari Blynk V12

// Blynk V12 = Start/Stop Button
BLYNK_WRITE(V12) {
  isStreaming = param.asInt();
}

float readFSR(int fsrPin) {
  int rawValue = analogRead(fsrPin);
  float voltage = rawValue * (3.3 / 4095.0);
  
  if (voltage < 0.01) return 0.0;  // Minimum detectable voltage
  
  float fsrResistance = (3.3 - voltage) * 10000.0 / voltage;
  if (fsrResistance > 600000.0) return 0.0;
  
  // More accurate force calculation with calibration constants
  float force = 1.0 / (fsrResistance / 1000000.0);  // Basic inverse relationship
  force = constrain(force, 0.0, 25.0);  // Limit to 25N max
  
  return force;
}

void updateFSRReadings() {
  // Remove the oldest reading
  fsrTotal1 -= fsrReadings1[readIndex];
  fsrTotal2 -= fsrReadings2[readIndex];
  fsrTotal3 -= fsrReadings3[readIndex];
  fsrTotal4 -= fsrReadings4[readIndex];

  // Add new readings
  fsrReadings1[readIndex] = readFSR(fsrPin1);
  fsrReadings2[readIndex] = readFSR(fsrPin2);
  fsrReadings3[readIndex] = readFSR(fsrPin3);
  fsrReadings4[readIndex] = readFSR(fsrPin4);

  fsrTotal1 += fsrReadings1[readIndex];
  fsrTotal2 += fsrReadings2[readIndex];
  fsrTotal3 += fsrReadings3[readIndex];
  fsrTotal4 += fsrReadings4[readIndex];

  // Calculate averages
  fsrForce1 = fsrTotal1 / numReadings;
  fsrForce2 = fsrTotal2 / numReadings;
  fsrForce3 = fsrTotal3 / numReadings;
  fsrForce4 = fsrTotal4 / numReadings;

  readIndex = (readIndex + 1) % numReadings;
}

void sendSensorData() {
  if (!isStreaming) return;

  // Update sensor IMU
  mySensor.accelUpdate();
  mySensor.gyroUpdate();
  
  // Update FSR readings with filtering
  updateFSRReadings();

  

  // Format data with 4 decimal places
  String dataString = 
    String(mySensor.accelX(), 4) + "," + String(mySensor.accelY(), 4) + "," + String(mySensor.accelZ(), 4) + "," +
    String(mySensor.gyroX(), 4) + "," + String(mySensor.gyroY(), 4) + "," + String(mySensor.gyroZ(), 4) + "," +
    String(fsrForce1, 4) + "," + String(fsrForce2, 4) + "," + String(fsrForce3, 4) + "," + String(fsrForce4, 4) + "," +
    gaitLabel;

  Serial.println(dataString);

  // TCP connection handling
  if (!tcpClient.connected()) {
    tcpClient.stop();
    if (!tcpClient.connect(server_ip_tcp, tcp_port)) {
      Serial.println("TCP connection failed");
      return;
    }
  }
  tcpClient.println(dataString);

  // Blynk updates with 4 decimal precision
  Blynk.virtualWrite(V1, String(fsrForce1, 4));
  Blynk.virtualWrite(V2, String(fsrForce2, 4));
  Blynk.virtualWrite(V3, String(fsrForce3, 4));
  Blynk.virtualWrite(V4, String(fsrForce4, 4));
  Blynk.virtualWrite(V5, String(mySensor.accelX(), 4));
  Blynk.virtualWrite(V6, String(mySensor.accelY(), 4));
  Blynk.virtualWrite(V7, String(mySensor.accelZ(), 4));
  Blynk.virtualWrite(V8, String(mySensor.gyroX(), 4));
  Blynk.virtualWrite(V9, String(mySensor.gyroY(), 4));
  Blynk.virtualWrite(V10, String(mySensor.gyroZ(), 4));
  Blynk.
  virtualWrite(V11, gaitLabel);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // Initialize FSR moving average arrays
  for (int i = 0; i < numReadings; i++) {
    fsrReadings1[i] = fsrReadings2[i] = fsrReadings3[i] = fsrReadings4[i] = 0;
  }

  mySensor.setWire(&Wire);
  mySensor.beginAccel();
  mySensor.beginGyro();

  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  Blynk.config(auth, server_ip_blynk, blynk_port);
  Blynk.connect();

  timer.setInterval(500L, sendSensorData); // 2Hz
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(ssid, pass);
    delay(1000);
    return;
  }

  Blynk.run();
  timer.run();
}