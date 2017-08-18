use glfw;
use glfw::Glfw;
use glfw::Context;
use glfw::Window;
use vulkano;
use vulkano::device::DeviceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDeviceType;
use vulkano::instance::debug::DebugCallback;
use vulkano::swapchain::Surface;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::ptr;

static ENABLE_VALIDATION_LAYERS: bool = true;

fn required_extensions(glfw: &Glfw) -> InstanceExtensions {
    let mut ideal = InstanceExtensions {
        khr_surface: true,
        khr_xlib_surface: true,
        khr_xcb_surface: true,
        khr_wayland_surface: true,
        khr_mir_surface: true,
        khr_android_surface: true,
        khr_win32_surface: true,
        mvk_ios_surface: true,
        mvk_macos_surface: true,
        ..InstanceExtensions::none()
    };

    println!("{:?}", ideal);

    let mut raw_extensions = glfw.get_required_instance_extensions().unwrap_or(vec![]);
    for ext in &raw_extensions {
        println!("extension needed from glfw: {}", ext);
    }


    if ENABLE_VALIDATION_LAYERS {
        ideal.ext_debug_report = true;
        raw_extensions.push("VK_EXT_debug_report".to_string())
    }

    match InstanceExtensions::supported_by_core() {
        Ok(supported) => supported.intersection(&ideal),
        Err(_) => InstanceExtensions::none(),
    }


    //TODO use raw extensions reported from glfw, not all of them at the same time
    //
    //    match InstanceExtensions::supported_by_core_raw() {
    //        Ok(supported) => supported.intersection(&raw_extensions),
    //        Err(_) => InstanceExtensions::none(),
    //    }
}

pub struct App {
    title: &'static str,
    width: u32,
    height: u32,
    glfw: Glfw,
    window: Window,
    vk_instance: Arc<Instance>,
    vk_debug_callback: Option<DebugCallback>,
    vk_surface: Arc<Surface>,
//    vk_physical_device: PhysicalDevice,
//    validation_layers: Vec<& 'static str>,
//    device_extensions: Vec<DeviceExtensions>,
}


impl App {
    pub fn new(width: u32, height: u32) -> Result<App, String> {
        let title = "Vulkan test";
        let (mut glfw, mut window) = App::init_window(width, height, title)?;
        let (mut instance, debug_callback, mut surface) = App::init_vulkan(&glfw, &window)?;

        Ok(App {
            title: title,
            width: width,
            height: height,
            glfw: glfw,
            window: window,
            vk_instance: instance,
            vk_debug_callback: debug_callback,
            vk_surface: surface,
//            vk_physical_device: physical_device,
//            validation_layers: vec!["VK_LAYER_LUNARG_standard_validation"],
//            device_extensions: vec![DeviceExtensions.khr_swapchain],
        })
    }

    pub fn run(mut self) -> Result<(), String> {
        self.main_loop()?;

        Ok(())
    }

    fn init_window(width: u32, height: u32, title: &str) -> Result<(Glfw, Window), String> {
        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).map_err(|e| format!("{}", e))?;
        glfw.window_hint(glfw::WindowHint::Visible(true));
        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
        glfw.window_hint(glfw::WindowHint::Resizable(false));

        match glfw.create_window(width, height, title, glfw::WindowMode::Windowed) {
            Some((mut window, _)) => {
//                window.make_current();
                Ok((glfw, window))
            }
            None => Err("Failed to create GLFW window.".to_string()),
        }
    }

    fn init_vulkan(glfw: &Glfw, window: &Window) -> Result<(Arc<Instance>, Option<DebugCallback>, Arc<Surface>), String> {
        let mut vk_instance = App::create_instance(glfw)?;
        let debug_callback = App::setup_debug_callback(&vk_instance)?;
        let mut surface = App::create_surface(&vk_instance, glfw, window)?;
        let mut physical_device = App::pick_physical_device(&vk_instance)?;
        App::create_logical_device()?;
        App::create_swap_chain()?;
        App::create_image_views()?;
        App::create_render_pass()?;
        App::create_graphics_pipeline()?;
        App::create_frame_buffers()?;
        App::create_command_pool()?;
        App::create_command_buffers()?;
        App::create_semaphores()?;

        Ok((vk_instance.clone(), debug_callback, surface))
    }

    fn create_instance(glfw: &Glfw) -> Result<Arc<Instance>, String> {
        let instance = {
            // When we create an instance, we have to pass a list of extensions that we want to enable.
            //
            // All the window-drawing functionalities are part of non-core extensions that we need
            // to enable manually. To do so, we ask the `vulkano_win` crate for the list of extensions
            // required to draw to a window.
            let extensions = required_extensions(glfw);

            // Now creating the instance.
            Instance::new(None, &extensions, None)
                .map_err(|e| format!("failed to create Vulkan instance: {}", e))?
        };

        Ok(instance)
    }

    fn setup_debug_callback(vk_instance: &Arc<Instance>) -> Result<Option<DebugCallback>, String> {
        if !ENABLE_VALIDATION_LAYERS {
            return Ok(None);
        }

        let _callback = DebugCallback::errors_and_warnings(&vk_instance, |msg| {
            println!("Debug callback: {:?}", msg.description);
        }).map_err(|e| format!("Error setting up vulkan debug callback: {}", e))?;

        Ok(Some(_callback))
    }

    fn create_surface(vk_instance: &Arc<Instance>, glfw: &Glfw, window: &Window) -> Result<Arc<Surface>, String> {
        use vulkano::VulkanObject;

        let surface: Arc<Surface> = unsafe {
            Arc::new(Surface::new( vk_instance.clone(), 0))
        };

        let result = unsafe {
            glfw::ffi::glfwCreateWindowSurface(vk_instance.internal_object(), window.window_ptr(), ptr::null_mut(), &mut surface.internal_object())
        };


        if result != 0 { // 0 is VK_SUCCESS
            return Err("Error creating window surface".to_string());
        }

        Ok(surface)
    }

    fn pick_physical_device(vk_instance: &Arc<Instance>) -> Result<PhysicalDevice, String> {
        let mut discrete_physical = None;
        let mut integrated_physical = None;
        let mut virtual_physical = None;
        let mut cpu_physical = None;
        let mut other_physical = None;

        for dev in vulkano::instance::PhysicalDevice::enumerate(vk_instance) {
            //TODO improve choice mechanism
            if dev.ty() == PhysicalDeviceType::DiscreteGpu { discrete_physical = Some(dev); }
            if dev.ty() == PhysicalDeviceType::IntegratedGpu { integrated_physical = Some(dev); }
            if dev.ty() == PhysicalDeviceType::VirtualGpu { virtual_physical = Some(dev); }
            if dev.ty() == PhysicalDeviceType::Cpu { cpu_physical = Some(dev); }
            if dev.ty() == PhysicalDeviceType::Other { other_physical = Some(dev); }
        };

        let mut physical;

        if discrete_physical.is_some() {
            physical = discrete_physical.unwrap();
        } else if integrated_physical.is_some() {
            physical = integrated_physical.unwrap();
            println!("Warning! Using an integrated graphics");
        } else if virtual_physical.is_some() {
            physical = virtual_physical.unwrap();
            println!("Warning! Virtual physical graphic device in use");
        } else if other_physical.is_some() {
            physical = other_physical.unwrap();
            println!("Warning! Type of physical device in use of unknown type");
        } else if cpu_physical.is_some() {
            physical = cpu_physical.unwrap();
            println!("Warning! Physical device in use is the CPU!!");
        } else {
            return Err("No supported physical device found".to_string());
        }

        // Some little debug infos.
        println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

        Ok(physical)
    }

    fn create_logical_device() -> Result<(), String> {
        Ok(())
    }

    fn create_swap_chain() -> Result<(), String> {
        Ok(())
    }

    fn create_image_views() -> Result<(), String> {
        Ok(())
    }

    fn create_render_pass() -> Result<(), String> {
        Ok(())
    }

    fn create_graphics_pipeline() -> Result<(), String> {
        Ok(())
    }

    fn create_frame_buffers() -> Result<(), String> {
        Ok(())
    }

    fn create_command_pool() -> Result<(), String> {
        Ok(())
    }

    fn create_command_buffers() -> Result<(), String> {
        Ok(())
    }

    fn create_semaphores() -> Result<(), String> {
        Ok(())
    }


    fn main_loop(mut self) -> Result<(), String> {
        for i in 0..10000000 {
            self.draw_frame();
        }

        Ok(())
    }

    fn draw_frame(&self) {
//        println!("New Frame!");
    }
}