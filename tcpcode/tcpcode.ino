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
int fsrPin1 = 32, fsrPin2 = 35, fsrPin3 = 33, fsrPin4 = 34;
int fsrValue1, fsrValue2, fsrValue3, fsrValue4;
float fsrForce1, fsrForce2, fsrForce3, fsrForce4;

bool isStreaming = false; // Kawalan dari Blynk V12

// Blynk V12 = Start/Stop Button
BLYNK_WRITE(V12) {
  int pinValue = param.asInt();
  isStreaming = (pinValue == 1);
}

void sendSensorData() {
  if (!isStreaming) return;

  // Update sensor IMU
  mySensor.accelUpdate();
  mySensor.gyroUpdate();

  // Baca FSR
  fsrValue1 = analogRead(fsrPin1);
  fsrValue2 = analogRead(fsrPin2);
  fsrValue3 = analogRead(fsrPin3);
  fsrValue4 = analogRead(fsrPin4);

  float voltage1 = fsrValue1 * (3.3 / 4095.0);
  float voltage2 = fsrValue2 * (3.3 / 4095.0);
  float voltage3 = fsrValue3 * (3.3 / 4095.0);
  float voltage4 = fsrValue4 * (3.3 / 4095.0);

  float fsrResistance1 = (voltage1 > 0) ? (3.3 - voltage1) * 10000.0 / voltage1 : 1000000;
  float fsrResistance2 = (voltage2 > 0) ? (3.3 - voltage2) * 10000.0 / voltage2 : 1000000;
  float fsrResistance3 = (voltage3 > 0) ? (3.3 - voltage3) * 10000.0 / voltage3 : 1000000;
  float fsrResistance4 = (voltage4 > 0) ? (3.3 - voltage4) * 10000.0 / voltage4 : 1000000;

  fsrForce1 = (fsrResistance1 <= 600000) ? 25.0 / (fsrResistance1 / 1000.0) : 0.0;
  fsrForce2 = (fsrResistance2 <= 600000) ? 25.0 / (fsrResistance2 / 1000.0) : 0.0;
  fsrForce3 = (fsrResistance3 <= 600000) ? 25.0 / (fsrResistance3 / 1000.0) : 0.0;
  fsrForce4 = (fsrResistance4 <= 600000) ? 25.0 / (fsrResistance4 / 1000.0) : 0.0;


  // Bentuk string data
  String dataString = String(mySensor.accelX()) + "," + String(mySensor.accelY()) + "," + String(mySensor.accelZ()) + "," +
                      String(mySensor.gyroX()) + "," + String(mySensor.gyroY()) + "," + String(mySensor.gyroZ()) + "," +
                      String(fsrForce1) + "," + String(fsrForce2) + "," + String(fsrForce3) + "," + String(fsrForce4) + "," +
   

  Serial.println(dataString); // Debug serial

  // Sambung TCP jika belum connected
  if (!tcpClient.connected()) {
    tcpClient.stop(); // buang sambungan lama
    if (tcpClient.connect(server_ip_tcp, tcp_port)) {
      Serial.println("✔️ TCP connected");
    } else {
      Serial.println("❌ Gagal sambung TCP");
      return;
    }
  }

  // Hantar data ke Python
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
}

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

  timer.setInterval(10L, sendSensorData); // 2Hz
}
void loop() {
  // Auto reconnect WiFi
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(ssid, pass);
    delay(1000);
    return;
  }

  Blynk.run();
  timer.run();
}