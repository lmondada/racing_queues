#[derive(Clone, Copy, PartialEq, Debug, Eq, PartialOrd, Ord)]
pub enum Entry<T> {
    Left(T),
    Right(T),
}

impl<T> Entry<T> {
    pub fn other_side(self) -> Entry<T> {
        match self {
            Entry::Left(v) => Entry::Right(v),
            Entry::Right(v) => Entry::Left(v),
        }
    }

    pub fn side(&self) -> Side {
        self.as_ref().map(|_| ())
    }

    pub fn as_ref(&self) -> Entry<&T> {
        match self {
            Entry::Left(v) => Entry::Left(v),
            Entry::Right(v) => Entry::Right(v),
        }
    }

    pub fn is_same_side<V>(&self, other: &Entry<V>) -> bool {
        self.side() == other.side()
    }

    pub fn is_other_side<V>(&self, other: &Entry<V>) -> bool {
        !self.is_same_side(other)
    }

    pub fn map<V, F: FnOnce(T) -> V>(self, f: F) -> Entry<V> {
        match self {
            Entry::Left(v) => Entry::Left(f(v)),
            Entry::Right(v) => Entry::Right(f(v)),
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Self::Left(_) => 0,
            Self::Right(_) => 1,
        }
    }

    pub fn value(&self) -> &T {
        match self {
            Self::Left(v) => v,
            Self::Right(v) => v,
        }
    }

    pub fn into_value(self) -> T {
        match self {
            Self::Left(v) => v,
            Self::Right(v) => v,
        }
    }
}

pub type Side = Entry<()>;

impl Side {
    pub fn entry<T>(&self, v: T) -> Entry<T> {
        self.map(|_| v)
    }

    pub fn from_index(u: usize) -> Self {
        match u {
            0 => Self::Left(()),
            1 => Self::Right(()),
            _ => panic!("Invalid index for Side"),
        }
    }
}
