use std::borrow::Cow;
use std::ffi::CStr;
use std::fs::File;
use std::io::Read;
use std::ptr;
use std::sync::Arc;

use glfw;
use glfw::Glfw;
use glfw::Context;
use glfw::Window;

use vulkano;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::cpu_access::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor::ShaderStages;
use vulkano::descriptor::descriptor::DescriptorDesc;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use vulkano::device::Device;
use vulkano::device::{Queue, QueuesIter};
use vulkano::format;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::SwapchainImage;
use vulkano::instance::PhysicalDevice;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDeviceType;
use vulkano::instance::debug::DebugCallback;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::shader::ShaderModule;
use vulkano::pipeline::shader::GraphicsShaderType;
use vulkano::pipeline::shader::{ShaderInterfaceDef,ShaderInterfaceDefEntry};
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::Surface;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::SurfaceTransform;
use vulkano::swapchain::PresentMode;
use vulkano::sync::GpuFuture;

static ENABLE_VALIDATION_LAYERS: bool = true;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 3],
}

impl_vertex!(Vertex, position, color);

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
    vk_device: Arc<Device>,
    vk_graphic_queue: Arc<Queue>,
    vk_swapchain: Arc<Swapchain>,
    vk_renderpass: Arc<RenderPassAbstract + Send + Sync>,
    vk_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    framebuffers: Vec<Arc<Framebuffer<Arc<RenderPassAbstract + Send + Sync>, ((), Arc<SwapchainImage>)>>>,
//    vk_physical_device: PhysicalDevice,
//    validation_layers: Vec<& 'static str>,
//    device_extensions: Vec<DeviceExtensions>,
}


impl App {
    pub fn new(width: u32, height: u32) -> Result<App, String> {
        let title = "Vulkan test";
        let (mut glfw, mut window) = App::init_window(width, height, title)?;
        let (mut instance, debug_callback, mut surface, mut device, mut graphic_queue, mut swapchain, mut render_pass, mut pipeline, mut vertex_buffer, mut framebuffers)
            = App::init_vulkan(&glfw, &window)?;

        Ok(App {
            title: title,
            width: width,
            height: height,
            glfw: glfw,
            window: window,
            vk_instance: instance,
            vk_debug_callback: debug_callback,
            vk_surface: surface,
            vk_device: device,
            vk_graphic_queue: graphic_queue,
            vk_swapchain: swapchain,
            vk_renderpass: render_pass,
            vk_pipeline: pipeline,
            vertex_buffer: vertex_buffer,
            framebuffers: framebuffers,
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

    fn init_vulkan(glfw: &Glfw, window: &Window)
        -> Result<(
            Arc<Instance>,
            Option<DebugCallback>,
            Arc<Surface>,
            Arc<Device>,
            Arc<Queue>,
            Arc<Swapchain>,
            Arc<RenderPassAbstract + Send + Sync>,
            Arc<GraphicsPipelineAbstract + Send + Sync>,
            Arc<CpuAccessibleBuffer<[Vertex]>>,
            Vec<Arc<Framebuffer<Arc<RenderPassAbstract + Send + Sync>, ((), Arc<SwapchainImage>)>>>,
        ), String>
    {
        let mut vk_instance = App::create_instance(glfw)?;
        let debug_callback = App::setup_debug_callback(&vk_instance)?;
        let mut surface = App::create_surface(&vk_instance, glfw, window)?;
        let mut physical_device = App::pick_physical_device(&vk_instance)?;
        let (mut device, mut queues) = App::create_logical_device(physical_device, &surface)?;

        // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
        // example we use only one queue, so we just retreive the first and only element of the
        // iterator and throw it away.
        let queue = queues.next().unwrap();

        let (mut swapchain, mut images) = App::create_swap_chain(window, physical_device, &surface, &device, queue.clone())?;

        App::create_image_views()?;
        let mut render_pass = App::create_render_pass(device.clone(), swapchain.clone())?;
        let mut pipeline = App::create_graphics_pipeline(device.clone(), swapchain.clone(), &images, render_pass.clone())?;
        let mut vertex_buffer = App::create_vertex_buffer(device.clone())?;
        let mut framebuffers = App::create_frame_buffers(images, render_pass.clone())?;
        App::create_command_pool()?;
        App::create_command_buffers()?;
        App::create_semaphores()?;

        Ok((vk_instance.clone(), debug_callback, surface, device, queue, swapchain, render_pass, pipeline, vertex_buffer, framebuffers))
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

        let mut _surface: u64 = 0;

        let result = unsafe {
            glfw::ffi::glfwCreateWindowSurface(vk_instance.internal_object(), window.window_ptr(), ptr::null_mut(), &mut _surface)
        };

        if result != 0 { // 0 is VK_SUCCESS
            return Err("Error creating window surface".to_string());
        }

        let surface: Arc<Surface> = unsafe {
            Arc::new(Surface::from_raw_surface( vk_instance.clone(), _surface))
        };

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

    fn create_logical_device(physical_device: PhysicalDevice, surface: &Surface) -> Result<(Arc<Device>, QueuesIter), String> {
        // The next step is to choose which GPU queue will execute our draw commands.
        //
        // Devices can provide multiple queues to run commands in parallel (for example a draw queue
        // and a compute queue), similar to CPU threads. This is something you have to have to manage
        // manually in Vulkan.
        //
        // In a real-life application, we would probably use at least a graphics queue and a transfers
        // queue to handle data transfers in parallel. In this example we only use one queue.
        //
        // We have to choose which queues to use early on, because we will need this info very soon.
        let queues = physical_device.queue_families().find(|&q| {
            // We take the first queue that supports drawing to our window.
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        });


        if queues.is_none() {
            return Err("couldn't find a graphical queue family".to_string());
        }

        // Now initializing the device. This is probably the most important object of Vulkan.
        //
        // We have to pass five parameters when creating a device:
        //
        // - Which physical device to connect to.
        //
        // - A list of optional features and extensions that our program needs to work correctly.
        //   Some parts of the Vulkan specs are optional and must be enabled manually at device
        //   creation. In this example the only thing we are going to need is the `khr_swapchain`
        //   extension that allows us to draw to a window.
        //
        // - A list of layers to enable. This is very niche, and you will usually pass `None`.
        //
        // - The list of queues that we are going to use. The exact parameter is an iterator whose
        //   items are `(Queue, f32)` where the floating-point represents the priority of the queue
        //   between 0.0 and 1.0. The priority of the queue is a hint to the implementation about how
        //   much it should prioritize queues between one another.
        //
        // The list of created queues is returned by the function alongside with the device.
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            .. vulkano::device::DeviceExtensions::none()
        };

        Device::new(physical_device, physical_device.supported_features(), &device_ext,
                    [(queues.unwrap(), 0.5)].iter().cloned())
            .map_err(|e| format!("failed to create device: {}", e))
    }

    fn create_swap_chain(window: &Window, physical_device: PhysicalDevice, surface: &Arc<Surface>, device: &Arc<Device>, queue: Arc<Queue>)
        -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), String>
    {
        // Querying the capabilities of the surface. When we create the swapchain we can only
        // pass values that are allowed by the capabilities.
        let caps = surface.capabilities(physical_device)
            .expect("failed to get surface capabilities");

        // We choose the dimensions of the swapchain to match the current dimensions of the window.
        // If `caps.current_extent` is `None`, this means that the window size will be determined
        // by the dimensions of the swapchain, in which case we just use the width and height defined above.
        let dimensions = caps.current_extent.unwrap_or([800, 600]);

        // The alpha mode indicates how the alpha value of the final image will behave. For example
        // you can choose whether the window will be opaque or transparent.
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Choosing the internal format that the images will have.
        let format = caps.supported_formats[0].0;

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
                       dimensions, 1, caps.supported_usage_flags, &queue,
                       SurfaceTransform::Identity, alpha, PresentMode::Fifo, true,
                       None).map_err(|e| format!("failed to create swapchain: {}", e))
    }

    fn create_image_views() -> Result<(), String> {
        Ok(())
    }

    fn create_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Result<Arc<RenderPassAbstract + Send + Sync>, String> {
        // CREATE RENDER PASS

        // The next step is to create a *render pass*, which is an object that describes where the
        // output of the graphics pipeline will go. It describes the layout of the images
        // where the colors, depth and/or stencil information will be written.
        Ok(Arc::new(single_pass_renderpass!(device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `load: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load: Clear,
                    // `store: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store: Store,
                    // `format: <ty>` indicates the type of the format of the image. This has to
                    // be one of the types of the `vulkano::format` module (or alternatively one
                    // of your structs that implements the `FormatDesc` trait). Here we use the
                    // generic `vulkano::format::Format` enum because we don't know the format in
                    // advance.
                    format: swapchain.format(),
                    // TODO:
                    samples: 1,
                }
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {}
            }
        ).map_err(|e| format!("failed to create render pass: {}", e))?))
    }

    fn create_graphics_pipeline(device: Arc<Device>, swapchain: Arc<Swapchain>, images: &Vec<Arc<SwapchainImage>>, render_pass: Arc<RenderPassAbstract + Send + Sync>)
        -> Result<Arc<GraphicsPipelineAbstract + Send + Sync>, String>
    {
        // The next step is to create the shaders.
        //
        // The raw shader creation API provided by the vulkano library is unsafe, for various reasons.
        //
        // TODO: explain this in details
        let vs_filepath = concat!(env!("OUT_DIR"), "/shader.vert.spv");
        let vs = {
            let mut f = File::open(vs_filepath).map_err(|e| format!("Error loading vertex shader: {} - Error: {}", vs_filepath, e))?;
            let mut v = vec![];
            f.read_to_end(&mut v).map_err(|e| format!("Impossible to read vertex shader file: {}", e))?;
            // Create a ShaderModule on a device the same Shader::load does it.
            // NOTE: You will have to verify correctness of the data by yourself!
            unsafe { ShaderModule::new(device.clone(), &v) }.map_err(|e| format!("Impossible to create vertex shader module: {}", e))?
        };

        let fs_filepath = concat!(env!("OUT_DIR"), "/shader.frag.spv");
        let fs = {
            let mut f = File::open(fs_filepath).map_err(|e| format!("Error loading fragment shader: {} - Error: {}", fs_filepath, e))?;
            let mut v = vec![];
            f.read_to_end(&mut v).map_err(|e| format!("Impossible to read fragment shader file: {}", e))?;
            unsafe { ShaderModule::new(device.clone(), &v) }.map_err(|e| format!("Impossible to create fragment shader module: {}", e))?
        };

        // This structure will tell Vulkan how input entries of our vertex shader
        // look like.
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        struct VertInput;
        unsafe impl ShaderInterfaceDef for VertInput {
            type Iter = VertInputIter;

            fn elements(&self) -> VertInputIter {
                VertInputIter(0)
            }
        }
        #[derive(Debug, Copy, Clone)]
        struct VertInputIter(u16);
        impl Iterator for VertInputIter {
            type Item = ShaderInterfaceDefEntry;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                // There are things to consider when giving out entries:
                // * There must be only one entry per one location, you can't have
                //   `color' and `position' entries both at 0..1 locations.  They also
                //   should not overlap.
                // * Format of each element must be no larger than 128 bits.
                if self.0 == 0 {
                    self.0 += 1;
                    return Some(ShaderInterfaceDefEntry {
                        location: 1..2,
                        format: format::Format::R32G32B32Sfloat,
                        name: Some(Cow::Borrowed("color"))
                    })
                }
                if self.0 == 1 {
                    self.0 += 1;
                    return Some(ShaderInterfaceDefEntry {
                        location: 0..1,
                        format: format::Format::R32G32Sfloat,
                        name: Some(Cow::Borrowed("position"))
                    })
                }
                None
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                // We must return exact number of entries left in iterator.
                let len = (2 - self.0) as usize;
                (len, Some(len))
            }
        }
        impl ExactSizeIterator for VertInputIter {
        }
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        struct VertOutput;
        unsafe impl ShaderInterfaceDef for VertOutput {
            type Iter = VertOutputIter;

            fn elements(&self) -> VertOutputIter {
                VertOutputIter(0)
            }
        }
        // This structure will tell Vulkan how output entries (those passed to next
        // stage) of our vertex shader look like.
        #[derive(Debug, Copy, Clone)]
        struct VertOutputIter(u16);
        impl Iterator for VertOutputIter {
            type Item = ShaderInterfaceDefEntry;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                if self.0 == 0 {
                    self.0 += 1;
                    return Some(ShaderInterfaceDefEntry {
                        location: 0..1,
                        format: format::Format::R32G32B32Sfloat,
                        name: Some(Cow::Borrowed("v_color"))
                    })
                }
                None
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = (1 - self.0) as usize;
                (len, Some(len))
            }
        }
        impl ExactSizeIterator for VertOutputIter {
        }
        // This structure describes layout of this stage.
        #[derive(Debug, Copy, Clone)]
        struct VertLayout(ShaderStages);
        unsafe impl PipelineLayoutDesc for VertLayout {
            // Number of descriptor sets it takes.
            fn num_sets(&self) -> usize { 1 }
            // Number of entries (bindings) in each set.
            fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
                match set { 0 => Some(1), _ => None, }
            }
            // Descriptor descriptions.
            fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
                match (set, binding) { _ => None, }
            }
            // Number of push constants ranges (think: number of push constants).
            fn num_push_constants_ranges(&self) -> usize { 0 }
            // Each push constant range in memory.
            fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
                if num != 0 || 0 == 0 { return None; }
                Some(PipelineLayoutDescPcRange { offset: 0,
                    size: 0,
                    stages: ShaderStages::all() })
            }
        }

        // Same as with our vertex shader, but for fragment one instead.
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        struct FragInput;
        unsafe impl ShaderInterfaceDef for FragInput {
            type Iter = FragInputIter;

            fn elements(&self) -> FragInputIter {
                FragInputIter(0)
            }
        }
        #[derive(Debug, Copy, Clone)]
        struct FragInputIter(u16);
        impl Iterator for FragInputIter {
            type Item = ShaderInterfaceDefEntry;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                if self.0 == 0 {
                    self.0 += 1;
                    return Some(ShaderInterfaceDefEntry {
                        location: 0..1,
                        format: format::Format::R32G32B32Sfloat,
                        name: Some(Cow::Borrowed("v_color"))
                    })
                }
                None
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = (1 - self.0) as usize;
                (len, Some(len))
            }
        }
        impl ExactSizeIterator for FragInputIter {
        }
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        struct FragOutput;
        unsafe impl ShaderInterfaceDef for FragOutput {
            type Iter = FragOutputIter;

            fn elements(&self) -> FragOutputIter {
                FragOutputIter(0)
            }
        }
        #[derive(Debug, Copy, Clone)]
        struct FragOutputIter(u16);
        impl Iterator for FragOutputIter {
            type Item = ShaderInterfaceDefEntry;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                // Note that color fragment color entry will be determined
                // automatically by Vulkano.
                if self.0 == 0 {
                    self.0 += 1;
                    return Some(ShaderInterfaceDefEntry {
                        location: 0..1,
                        format: format::Format::R32G32B32A32Sfloat,
                        name: Some(Cow::Borrowed("f_color"))
                    })
                }
                None
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = (1 - self.0) as usize;
                (len, Some(len))
            }
        }
        impl ExactSizeIterator for FragOutputIter {
        }
        // Layout same as with vertex shader.
        #[derive(Debug, Copy, Clone)]
        struct FragLayout(ShaderStages);
        unsafe impl PipelineLayoutDesc for FragLayout {
            fn num_sets(&self) -> usize { 1 }
            fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
                match set { 0 => Some(1), _ => None, }
            }
            fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
                match (set, binding) { _ => None, }
            }
            fn num_push_constants_ranges(&self) -> usize { 0 }
            fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
                if num != 0 || 0 == 0 { return None; }
                Some(PipelineLayoutDescPcRange { offset: 0,
                    size: 0,
                    stages: ShaderStages::all() })
            }
        }

        // NOTE: ShaderModule::*_shader_entry_point calls do not do any error
        // checking and you have to verify correctness of what you are doing by
        // yourself.
        //
        // You must be extra careful to specify correct entry point, or program will
        // crash at runtime outside of rust and you will get NO meaningful error
        // information!
        let vert_main = unsafe { vs.graphics_entry_point(
            CStr::from_bytes_with_nul_unchecked(b"main\0"),
            VertInput,
            VertOutput,
            VertLayout(ShaderStages { vertex: true, ..ShaderStages::none() }),
            GraphicsShaderType::Vertex
        ) };

        let frag_main = unsafe { fs.graphics_entry_point(
            CStr::from_bytes_with_nul_unchecked(b"main\0"),
            FragInput,
            FragOutput,
            FragLayout(ShaderStages { fragment: true, ..ShaderStages::none() }),
            GraphicsShaderType::Fragment
        ) };

        let sub_pass = Subpass::from(render_pass, 0);
        let sub_pass = if sub_pass.is_some() {
            sub_pass.unwrap()
        } else {
            return Err("Impossible to create subpass from renderpass".to_string());
        };

        // Before we draw we have to create what is called a pipeline. This is similar to an OpenGL
        // program, but much more specific.
        let pipeline = Arc::new(GraphicsPipeline::start()
            // We need to indicate the layout of the vertices.
            // The type `SingleBufferDefinition` actually contains a template parameter corresponding
            // to the type of each vertex. But in this code it is automatically inferred.
            .vertex_input(SingleBufferDefinition::<Vertex>::new())
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one. The `main` word of `main_entry_point` actually corresponds to the name of
            // the entry point.
            .vertex_shader(vert_main, ())
            // The content of the vertex buffer describes a list of triangles.
            .triangle_list()
            .viewports([
                Viewport {
                    origin: [0.0, 0.0],
                    depth_range: 0.0..1.0,
                    dimensions: [
                        images[0].dimensions()[0] as f32,
                        images[0].dimensions()[1] as f32,
                    ],
                },
            ].iter().cloned())
            // See `vertex_shader`.
            .fragment_shader(frag_main, ())
            .cull_mode_front()
            .front_face_counter_clockwise()
            .depth_stencil_disabled()
            // We have to indicate which subpass of which render pass this pipeline is going to be used
            // in. The pipeline will only be usable from this particular subpass.
            .render_pass(sub_pass)
            // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
            .build(device.clone())
            .map_err(|e| format!("Error creating the pipeline: {}", e))?);

        Ok(pipeline)
    }

    fn create_vertex_buffer(device: Arc<Device>) -> Result<Arc<CpuAccessibleBuffer<[Vertex]>>, String> {
        CpuAccessibleBuffer::from_iter(
            device,
            BufferUsage::all(),
            [
                Vertex { position: [-1.0,  1.0], color: [1.0, 0.0, 0.0] },
                Vertex { position: [ 0.0, -1.0], color: [1.0, 0.0, 0.0] },
                Vertex { position: [ 1.0,  1.0], color: [1.0, 0.0, 0.0] },
            ].iter().cloned()
        ).map_err(|e| format!("Failed to create Vertex Buffers: {}", e))
    }

    fn create_frame_buffers(images: Vec<Arc<SwapchainImage>>,render_pass: Arc<RenderPassAbstract + Send + Sync>)
        -> Result<Vec<Arc<Framebuffer<Arc<RenderPassAbstract + Send + Sync>, ((), Arc<SwapchainImage>)>>>, String>
    {
        let framebuffers: Vec<_> = images
            .iter()
            .map(|image| Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone()).unwrap()
                    .build().unwrap(),
            ))
            .collect();

        Ok(framebuffers)
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
        while !self.window.should_close() {
            self.glfw.poll_events();
            self.draw_frame();
        }

        Ok(())
    }

    fn draw_frame(&self) {
//        println!("New Frame!");
        let (image_num, acquire_future) = vulkano::swapchain::acquire_next_image(
            self.vk_swapchain.clone(),
            None,
        ).expect("failed to acquire swapchain in time");

        let command_buffer = AutoCommandBufferBuilder::new(
            self.vk_device.clone(),
            self.vk_graphic_queue.family(),
        ).unwrap();

        let command_buffer = command_buffer.begin_render_pass(
            self.framebuffers[image_num].clone(),
            false,
            vec![[0.0, 0.0, 0.0, 1.0].into(), 1.0.into()],
        ).unwrap();

        let command_buffer = command_buffer.draw(
            self.vk_pipeline.clone(),
            DynamicState::none(),
            vec![self.vertex_buffer.clone()],
            (),
            (),
        ).unwrap();

        let command_buffer = command_buffer.end_render_pass().unwrap();

        let command_buffer = command_buffer.build().unwrap();

        acquire_future
            .then_execute(self.vk_graphic_queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(self.vk_graphic_queue.clone(), self.vk_swapchain.clone(), image_num)
            .then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();
    }
}
