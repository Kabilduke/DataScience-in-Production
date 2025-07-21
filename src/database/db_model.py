from sqlalchemy import REAL, INTEGER, VARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.config import settings


class Base(DeclarativeBase):
    pass  # noqa: WPS420


class classification(Base):
    __tablename__ = settings.table_name

    Column1: Mapped[int] = mapped_column(INTEGER())
    Age: Mapped[int] = mapped_column(INTEGER(), primary_key=True)
    Gender: Mapped[str] = mapped_column(VARCHAR())
    BMI: Mapped[int] = mapped_column(INTEGER())
    Chol: Mapped[float] = mapped_column(REAL())
    TG: Mapped[float] = mapped_column(REAL())
    HDL: Mapped[float] = mapped_column(REAL())
    LDL: Mapped[float] = mapped_column(REAL())
    Cr: Mapped[float] = mapped_column(REAL())
    BUN: Mapped[float] = mapped_column(REAL())
    Diagnosis: Mapped[int] = mapped_column(INTEGER())



