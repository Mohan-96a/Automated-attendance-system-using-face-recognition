"""
Pydantic Schemas for Academic Management: Departments, Subjects, Classrooms, Timetables.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# --- Department Schemas ---
class DepartmentCreateRequest(BaseModel):
    id: str = Field(description="e.g. DEP-CS")
    name: str
    code: str
    head: Optional[str] = None


class DepartmentResponse(BaseModel):
    id: str
    name: str
    code: str
    head: Optional[str] = None


# --- Classroom Schemas ---
class GeofenceModel(BaseModel):
    latitude: float
    longitude: float
    radius_meters: float = 50.0


class ClassroomCreateRequest(BaseModel):
    id: str = Field(description="e.g. CR-101")
    name: str
    block: str = "Main Block"
    capacity: int = 60
    geofence: Optional[GeofenceModel] = None


class ClassroomResponse(BaseModel):
    id: str
    name: str
    block: str
    capacity: int
    geofence: Optional[GeofenceModel] = None


# --- Subject Schemas ---
class SubjectCreateRequest(BaseModel):
    id: str = Field(description="e.g. SUB-CS301")
    name: str
    code: str
    department_id: str
    semester: int = 1
    credits: int = 3


class SubjectResponse(BaseModel):
    id: str
    name: str
    code: str
    department_id: str
    semester: int
    credits: int


# --- Timetable Schemas ---
class TimetableCreateRequest(BaseModel):
    subject_id: str
    classroom_id: str
    faculty_id: str
    day_of_week: str  # Monday, Tuesday...
    start_time: str   # "09:00"
    end_time: str     # "10:00"
    section: str = "A"


class TimetableResponse(BaseModel):
    id: str
    subject_id: str
    subject_name: Optional[str] = None
    faculty_id: str
    faculty_name: Optional[str] = None
    classroom_id: str
    classroom_name: Optional[str] = None
    day_of_week: str
    start_time: str
    end_time: str
    section: str
