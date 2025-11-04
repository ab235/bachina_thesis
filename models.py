from sqlalchemy import Column, Integer, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from db import Base, engine

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(1536))
    document = relationship("Document", back_populates="chunks")
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_doc_ord"), 
    )#unique chunks

def create_all():
    Base.metadata.create_all(bind=engine)
