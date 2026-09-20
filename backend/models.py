from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    func
)

from sqlalchemy.dialects.postgresql import JSONB

from database import Base


class Character(Base):
    __tablename__ = "characters"

    id = Column(Integer, primary_key=True)

    character_name = Column(
        String(150),
        nullable=False
    )

    universe = Column(
        String(150),
        nullable=False
    )

    image_url = Column(Text)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "character_name",
            "universe",
            name="unique_character"
        ),
    )


class GameResult(Base):
    __tablename__ = "game_results"

    id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True
    )

    character_id = Column(
        Integer,
        ForeignKey(
            "characters.id",
            ondelete="RESTRICT"
        ),
        nullable=False
    )

    match_percentage = Column(
        Integer
    )

    stats = Column(
        JSONB
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )