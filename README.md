# SmartAttend AI 🎓

**AI-Powered Smart Attendance Management System**

SmartAttend AI is a full-stack intelligent attendance platform that combines **facial recognition**, **anti-spoofing**, and **real-time computer vision** with a modern web application. It automates classroom attendance marking with separate dashboards for administrators, faculty, and students.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 Face Detection | Real-time multi-face detection with quality & angle assessment |
| 🧠 ArcFace Recognition | Accurate CPU-optimised face recognition via InsightFace |
| 🛡️ Anti-Spoofing | Liveness detection — blocks printed photos & screen replays |
| 📡 Real-Time Streaming | WebSocket-based live camera feed with attendance updates |
| 🏛️ Role-Based Access | Separate dashboards for Admin, Faculty & Students |
| 📊 Analytics | Attendance trends, risk predictions, and automated reports |
| 📄 Report Export | Excel (XLSX) and PDF attendance reports |
| 🔐 Secure Auth | JWT authentication + PBKDF2 password hashing |
| 🗄️ Zero-Config DB | Works without MongoDB — built-in JSON fallback engine |

---

## 🏗️ Architecture

```
SmartAttend AI
├── ai_engine/          # Phase 1: Computer Vision & Face Recognition
│   ├── config/         # AI thresholds, camera config
│   ├── detection/      # Face detection, landmarks, quality
│   ├── recognition/    # ArcFace embeddings, face matching
│   ├── tracking/       # Cross-frame face tracking
│   ├── anti_spoofing/  # Liveness detection
│   ├── enrollment/     # Face registration pipeline
│   ├── evaluation/     # Benchmark & performance tests
│   └── engine.py       # Unified inference pipeline
│
├── backend/            # Phase 2: FastAPI Backend
│   └── app/
│       ├── config.py         # App settings (env-based)
│       ├── main.py           # FastAPI app entry point
│       ├── database/         # MongoDB + JSON fallback
│       ├── routes/           # REST API route handlers
│       ├── services/         # Business logic layer
│       ├── schemas/          # Pydantic request/response models
│       ├── middleware/       # JWT auth, RBAC
│       └── utils/            # Security utilities
│
├── frontend/           # Phase 3: React + Tailwind UI (coming soon)
├── ml_analytics/       # Phase 4: ML predictions, reports (coming soon)
├── tests/              # Pytest test suite
├── docs/               # Documentation
├── docker/             # Docker configuration (Phase 5)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | React.js, Tailwind CSS, Shadcn/UI, Recharts |
| **Backend** | Python FastAPI, Uvicorn |
| **Face Detection** | OpenCV (Haar + DNN) |
| **Face Recognition** | InsightFace / ArcFace (ONNX, CPU) |
| **Anti-Spoofing** | FFT texture analysis, micro-gradient liveness |
| **Face Tracking** | OpenCV centroid + IoU tracker |
| **Machine Learning** | Scikit-learn |
| **Database** | MongoDB (Motor) + JSON fallback engine |
| **Authentication** | JWT (python-jose), PBKDF2+salt |
| **Real-Time** | WebSockets (FastAPI native) |
| **Reports** | OpenPyXL (Excel), ReportLab (PDF) |
| **Deployment** | Docker, Cloud hosting |

**Hardware:** Intel i5 CPU, 16 GB RAM — **no GPU required**.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend — Phase 3)
- MongoDB (optional — JSON fallback works without it)

### 1. Clone & Install

```bash
git clone https://github.com/Mohan-96a/Automated-attendance-system-using-face-recognition.git
cd Automated-attendance-system-using-face-recognition

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — set SECRET_KEY, optionally MONGODB_URI
```

### 3. Start the Backend

```bash
# From project root
uvicorn backend.app.main:app --reload --port 8000
```

The API will auto-seed demo accounts on first run.

### 4. Open API Docs

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 🔑 Default Demo Credentials

| Role | Email | Password |
|---|---|---|
| Admin | `admin@smartattend.ai` | `admin123` |
| Faculty | `faculty@smartattend.ai` | `faculty123` |
| Student | `student@smartattend.ai` | `student123` |

> ⚠️ Change all credentials before any production deployment.

---

## 📡 API Endpoints

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/login` | Login (returns JWT tokens) |
| POST | `/api/v1/auth/refresh` | Refresh access token |
| POST | `/api/v1/auth/logout` | Logout |
| POST | `/api/v1/auth/reset-password` | Request password reset |

### Students
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/students/` | List all students |
| POST | `/api/v1/students/` | Register student |
| GET | `/api/v1/students/{id}` | Get student profile |
| PUT | `/api/v1/students/{id}` | Update student |
| DELETE | `/api/v1/students/{id}` | Delete student |
| POST | `/api/v1/students/bulk-import` | Bulk CSV/Excel import |
| POST | `/api/v1/students/{id}/enroll-face` | Register face biometric |

### Attendance
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/attendance/sessions/` | Create session |
| POST | `/api/v1/attendance/sessions/{id}/start` | Start session |
| POST | `/api/v1/attendance/sessions/{id}/stop` | Stop session |
| POST | `/api/v1/attendance/sessions/{id}/process-frame` | Process camera frame |
| GET | `/api/v1/attendance/records/` | Attendance history |
| POST | `/api/v1/attendance/manual` | Manual attendance |
| WS | `/ws/attendance/{session_id}` | Live WebSocket stream |

### Faculty & Academic
| Method | Endpoint | Prefix |
|---|---|---|
| CRUD | `/api/v1/faculty/` | Faculty management |
| CRUD | `/api/v1/academic/departments/` | Departments |
| CRUD | `/api/v1/academic/subjects/` | Subjects |
| CRUD | `/api/v1/academic/classrooms/` | Classrooms |
| CRUD | `/api/v1/academic/timetables/` | Timetables |

---

## 🧪 Running Tests

```bash
# Backend tests (runs with JSON fallback, no MongoDB needed)
FORCE_JSON_DB=true python -m pytest tests/test_backend.py -v

# AI engine benchmark
python -m ai_engine.evaluation.evaluate_models
```

---

## 📋 Five-Phase Development Plan

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | AI Engine & Computer Vision | ✅ Complete |
| **Phase 2** | Backend, Database & Authentication | ✅ Complete |
| **Phase 3** | Full-Stack Frontend & Dashboards | 🔧 In Progress |
| **Phase 4** | Smart Analytics, ML & Automation | ⏳ Planned |
| **Phase 5** | Testing, Optimization & Deployment | ⏳ Planned |

---

## 🗄️ Database Collections

| Collection | Purpose |
|---|---|
| `users` | User accounts and roles |
| `students` | Student profiles |
| `faculty` | Faculty profiles |
| `departments` | Academic departments |
| `subjects` | Subject / course records |
| `classrooms` | Room information |
| `timetables` | Class schedules |
| `face_embeddings` | Encrypted biometric templates |
| `attendance_sessions` | Active and past sessions |
| `attendance_records` | Individual attendance records |
| `notifications` | User notifications |
| `correction_requests` | Attendance dispute requests |
| `audit_logs` | Full security audit trail |
| `system_settings` | Application configuration |

---

## 🔒 Security

- **Biometric data** is AES-encrypted (Fernet) before storage
- **Passwords** are hashed with PBKDF2-SHA256 + unique salt
- **JWT tokens** with configurable expiry and refresh rotation
- **RBAC** — every endpoint enforces Admin / Faculty / Student roles
- **Rate limiting** and **input validation** on all API routes
- **Audit logging** — all sensitive operations are recorded

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'feat: add my feature'`
4. Push and open a Pull Request

---

## 📄 License

This project is developed for academic and educational purposes.

---

*SmartAttend AI — Automating attendance, one face at a time.*
