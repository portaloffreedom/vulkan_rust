mod actor_frame;
pub use self::actor_frame::ActorFrame;

use objects::Object;

pub struct Actor<T: Sized> {
    frame: ActorFrame,
    object: Option<Object<[T]>>,
    children_frame: Vec<Actor<T>>,
}