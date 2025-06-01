#define BLYNK_PRINT Serial
#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
#include <Wire.h>
#include <MPU9250_asukiaaa.h>

// ========== BLYNK Legacy ==========
char auth[] = "s6twEQqusbfwEcrQYdID2avZ00GTA_r8";
char ssid[] = "Adawiyah";
char pass[] = "12345678";

// Blynk server
IPAddress server_ip_blynk(152, 42, 239, 200);
int blynk_port = 8080;

// ========== TCP Server (Python) ==========
IPAddress server_ip_tcp(172, 20, 10, 10); // IP PC Python
int tcp_port = 3000;
WiFiClient tcpClient;

// ========== Sensor ==========
MPU9250_asukiaaa mySensor;
int fsrPin1 = 32, fsrPin2 = 35, fsrPin3 = 33, fsrPin4 = 34;
float fsrForce1, fsrForce2, fsrForce3, fsrForce4;

// ========== Kawalan ==========
BlynkTimer timer;
bool isStreaming = false;
String gaitLabel = "Normal";

// ========== Kawalan Blynk ==========
BLYNK_WRITE(V12) {
  isStreaming = (param.asInt() == 1);
}

// ========== Hantar Data ==========
void sendSensorData() {
  if (!isStreaming) return;

  // IMU update
  mySensor.accelUpdate();
  mySensor.gyroUpdate();

  // FSR baca analog
  int val1 = analogRead(fsrPin1);
  int val2 = analogRead(fsrPin2);
  int val3 = analogRead(fsrPin3);
  int val4 = analogRead(fsrPin4);

  float v1 = val1 * (3.3 / 4095.0);
  float v2 = val2 * (3.3 / 4095.0);
  float v3 = val3 * (3.3 / 4095.0);
  float v4 = val4 * (3.3 / 4095.0);

  float r1 = (v1 > 0) ? (3.3 - v1) * 10000.0 / v1 : 1000000;
  float r2 = (v2 > 0) ? (3.3 - v2) * 10000.0 / v2 : 1000000;
  float r3 = (v3 > 0) ? (3.3 - v3) * 10000.0 / v3 : 1000000;
  float r4 = (v4 > 0) ? (3.3 - v4) * 10000.0 / v4 : 1000000;

  fsrForce1 = (r1 <= 600000) ? 25.0 / (r1 / 1000.0) : 0.0;
  fsrForce2 = (r2 <= 600000) ? 25.0 / (r2 / 1000.0) : 0.0;
  fsrForce3 = (r3 <= 600000) ? 25.0 / (r3 / 1000.0) : 0.0;
  fsrForce4 = (r4 <= 600000) ? 25.0 / (r4 / 1000.0) : 0.0;

  // Gait classification (sementara)
  if (fsrForce1 > 20 && fsrForce2 > 20 && fsrForce3 > 20 || fsrForce4 > 0.20) {
    gaitLabel = "Abnormal";
  } else {
    gaitLabel = "Normal";
  }

  String dataString = String(mySensor.accelX()) + "," + String(mySensor.accelY()) + "," + String(mySensor.accelZ()) + "," +
                      String(mySensor.gyroX()) + "," + String(mySensor.gyroY()) + "," + String(mySensor.gyroZ()) + "," +
                      String(fsrForce1) + "," + String(fsrForce2) + "," + String(fsrForce3) + "," + String(fsrForce4) + "," +
                      gaitLabel;

  Serial.println(dataString);

  // Cuba sambung TCP kalau belum connected
  if (!tcpClient.connected()) {
    tcpClient.stop();
    if (tcpClient.connect(server_ip_tcp, tcp_port)) {
      Serial.println("✔️ Reconnected to TCP");
    } else {
      Serial.println("❌ Gagal sambung TCP");
      return;
    }
  }

  // Hantar ke Python
  tcpClient.println(dataString);

  // Blynk display
  Blynk.virtualWrite(V1, fsrForce1);
  Blynk.virtualWrite(V2, fsrForce2);
  Blynk.virtualWrite(V3, fsrForce3);
  Blynk.virtualWrite(V4, fsrForce4);
  Blynk.virtualWrite(V5, mySensor.accelX());
  Blynk.virtualWrite(V6, mySensor.accelY());
  Blynk.virtualWrite(V7, mySensor.accelZ());
  Blynk.virtualWrite(V8, mySensor.gyroX());
  Blynk.virtualWrite(V9, mySensor.gyroY());
  Blynk.virtualWrite(V10, mySensor.gyroZ());
  Blynk.virtualWrite(V11, gaitLabel);
}

// ========== Setup ==========
void setup() {
  Serial.begin(9600);
  Wire.begin();

  pinMode(fsrPin1, INPUT);
  pinMode(fsrPin2, INPUT);
  pinMode(fsrPin3, INPUT);
  pinMode(fsrPin4, INPUT);

  mySensor.setWire(&Wire);
  mySensor.beginAccel();
  mySensor.beginGyro();

  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("✔️ WiFi connected");

  Blynk.config(auth, server_ip_blynk, blynk_port);
  Blynk.connect();

  timer.setInterval(500L, sendSensorData); // 2 Hz
}

// ========== Loop ==========
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(ssid, pass);
    delay(1000);
    return;
  }

  Blynk.run();
  timer.run();
}