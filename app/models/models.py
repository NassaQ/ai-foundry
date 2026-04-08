from typing import Optional
import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKeyConstraint,
    Identity,
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
        PrimaryKeyConstraint("doc_id", name="PK__Document__8AD02924828124C8"),
    )

    doc_id: Mapped[int] = mapped_column(BigInteger, Identity(start=1, increment=1), primary_key=True)
    filename: Mapped[str] = mapped_column(Unicode(255, "SQL_Latin1_General_CP1_CI_AS"), nullable=False)
    mongo_doc_id: Mapped[str] = mapped_column(String(36, "SQL_Latin1_General_CP1_CI_AS"), nullable=False)

    Processing_Status: Mapped[list["ProcessingStatus"]] = relationship("ProcessingStatus", back_populates="doc")


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
