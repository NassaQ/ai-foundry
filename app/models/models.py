from typing import Optional
import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Identity,
    Integer,
    PrimaryKeyConstraint,
    String,
    Unicode,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Documents(Base):
    __tablename__ = "Documents"
    __table_args__ = (
        ForeignKeyConstraint(["path_id"], ["Virtual_Paths.path_id"], name="FK_Doc_Path"),
        ForeignKeyConstraint(["uploaded_by_user_id"], ["Users.user_id"], name="FK_Doc_Uploader"),
        PrimaryKeyConstraint("doc_id", name="PK__Document__8AD02924828124C8"),
    )

    doc_id: Mapped[int] = mapped_column(BigInteger, Identity(start=1, increment=1), primary_key=True)
    filename: Mapped[str] = mapped_column(Unicode(255, "SQL_Latin1_General_CP1_CI_AS"), nullable=False)
    path_id: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_by_user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    mongo_doc_id: Mapped[str] = mapped_column(String(36, "SQL_Latin1_General_CP1_CI_AS"), nullable=False)
    uploaded_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, server_default=text("(getdate())"))
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, onupdate=text("(getutcdate())"))
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger)
    content_type: Mapped[Optional[str]] = mapped_column(String(100, "SQL_Latin1_General_CP1_CI_AS"))
    file_type: Mapped[Optional[str]] = mapped_column(String(10, "SQL_Latin1_General_CP1_CI_AS"))

    Processing_Status: Mapped[list["ProcessingStatus"]] = relationship("ProcessingStatus", back_populates="doc")
    Ocr_Results: Mapped[list["OcrResult"]] = relationship("OcrResult", back_populates="doc")


class ProcessingStatus(Base):
    __tablename__ = "Processing_Status"
    __table_args__ = (
        ForeignKeyConstraint(
            ["doc_id"], ["Documents.doc_id"], name="FK_ProcStatus_Doc"
        ),
        PrimaryKeyConstraint("status_id", name="PK__Processi__3683B5310CA4907C"),
    )

    status_id: Mapped[int] = mapped_column(BigInteger, Identity(start=1, increment=1), primary_key=True)
    doc_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    stage_name: Mapped[str] = mapped_column(String(50, "SQL_Latin1_General_CP1_CI_AS"), nullable=False)
    status: Mapped[str] = mapped_column(String(20, "SQL_Latin1_General_CP1_CI_AS"), nullable=False)
    start_time: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, server_default=text("(getdate())"))
    end_time: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    error_message: Mapped[Optional[str]] = mapped_column(Unicode(collation="SQL_Latin1_General_CP1_CI_AS"))

    doc: Mapped["Documents"] = relationship(
        "Documents", back_populates="Processing_Status"
    )


class OcrResult(Base):
    __tablename__ = "Ocr_Results"
    __table_args__ = (
        ForeignKeyConstraint(
            ["doc_id"], ["Documents.doc_id"], name="FK_OcrResults_Doc"
        ),
        PrimaryKeyConstraint("result_id", name="PK_OcrResults"),
    )

    result_id: Mapped[int] = mapped_column(BigInteger, Identity(start=1, increment=1), primary_key=True)
    doc_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    primary_language: Mapped[str] = mapped_column(String(10, "SQL_Latin1_General_CP1_CI_AS"), nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(50, "SQL_Latin1_General_CP1_CI_AS"))
    classification_confidence: Mapped[Optional[float]] = mapped_column(Float)
    cost_usd_ocr: Mapped[float] = mapped_column(Float, nullable=False)
    cost_usd_classification: Mapped[Optional[float]] = mapped_column(Float)
    processed_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, server_default=text("(getutcdate())"))

    doc: Mapped["Documents"] = relationship(
        "Documents", back_populates="Ocr_Results"
    )
