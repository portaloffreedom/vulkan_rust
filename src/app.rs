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
use vulkano::command_buffer::AutoCommandBuffer;
use vulkano::command_buffer::DynamicState;
use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc;
use vulkano::device::Device;
use vulkano::device::{Queue, QueuesIter};
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::SwapchainImage;
use vulkano::image::ImmutableImage;
use vulkano::instance::PhysicalDevice;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDeviceType;
use vulkano::instance::debug::DebugCallback;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::Surface;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::SurfaceTransform;
use vulkano::swapchain::PresentMode;
use vulkano::sync::GpuFuture;
use vulkano::sync::NowFuture;

use shaders::Shader;
use shaders::Vertex;

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
    vk_device: Arc<Device>,
    vk_graphic_queue: Arc<Queue>,
    vk_swapchain: Arc<Swapchain>,
    vk_renderpass: Arc<RenderPassAbstract + Send + Sync>,
    vk_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    framebuffers: Vec<Arc<Framebuffer<Arc<RenderPassAbstract + Send + Sync>, ((), Arc<SwapchainImage>)>>>,
    command_buffers: Vec<Arc<AutoCommandBuffer<StandardCommandPoolAlloc>>>,
//    vk_physical_device: PhysicalDevice,
//    validation_layers: Vec<& 'static str>,
//    device_extensions: Vec<DeviceExtensions>,
}


impl App {
    pub fn new(width: u32, height: u32) -> Result<App, String> {
        let title = "Vulkan test";
        let (glfw, window) = App::init_window(width, height, title)?;
        let (instance, debug_callback, surface, device, graphic_queue, swapchain, render_pass, pipeline, vertex_buffer, framebuffers, command_buffers )
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
            command_buffers: command_buffers,
//            vk_physical_device: physical_device,
//            validation_layers: vec!["VK_LAYER_LUNARG_standard_validation"],
//            device_extensions: vec![DeviceExtensions.khr_swapchain],
        })
    }

    pub fn run(self) -> Result<(), String> {
        self.main_loop()?;

        Ok(())
    }

    fn init_window(width: u32, height: u32, title: &str) -> Result<(Glfw, Window), String> {
        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).map_err(|e| format!("{}", e))?;
        glfw.window_hint(glfw::WindowHint::Visible(true));
        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
        glfw.window_hint(glfw::WindowHint::Resizable(false));

        match glfw.create_window(width, height, title, glfw::WindowMode::Windowed) {
            Some((window, _)) => {
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
            Vec<Arc<AutoCommandBuffer<StandardCommandPoolAlloc>>>,
        ), String>
    {
        let vk_instance = App::create_instance(glfw)?;
        let debug_callback = App::setup_debug_callback(&vk_instance)?;
        let surface = App::create_surface(&vk_instance, window)?;
        let physical_device = App::pick_physical_device(&vk_instance)?;
        let (device, mut queues) = App::create_logical_device(physical_device, &surface)?;

        // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
        // example we use only one queue, so we just retreive the first and only element of the
        // iterator and throw it away.
        let graphic_queue = queues.next().unwrap();

        let (swapchain, images) = App::create_swap_chain(physical_device, &surface, &device, graphic_queue.clone())?;

        App::create_image_views()?;
        let render_pass = App::create_render_pass(device.clone(), swapchain.clone())?;
        let pipeline = App::create_graphics_pipeline(device.clone(), &images, render_pass.clone())?;
        let vertex_buffer = App::create_vertex_buffer(device.clone())?;
        let (texture, tex_future) = App::load_and_create_texture_buffer(&graphic_queue)?;
        let framebuffers = App::create_frame_buffers(images, render_pass.clone())?;
        App::create_command_pool()?;
        let command_buffers = App::create_command_buffers(&device, &graphic_queue, &pipeline, &vertex_buffer, &texture, &framebuffers)?;
        App::create_semaphores()?;

        Ok((vk_instance.clone(), debug_callback, surface, device, graphic_queue, swapchain, render_pass, pipeline, vertex_buffer, framebuffers, command_buffers))
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

    fn create_surface(vk_instance: &Arc<Instance>, window: &Window) -> Result<Arc<Surface>, String> {
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

        let physical;

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

    fn create_swap_chain(physical_device: PhysicalDevice, surface: &Arc<Surface>, device: &Arc<Device>, queue: Arc<Queue>)
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

    fn create_graphics_pipeline(device: Arc<Device>, images: &Vec<Arc<SwapchainImage>>, render_pass: Arc<RenderPassAbstract + Send + Sync>)
        -> Result<Arc<GraphicsPipelineAbstract + Send + Sync>, String>
    {
        // The next step is to create the shaders.
        //
        // The raw shader creation API provided by the vulkano library is unsafe, for various reasons.
        //
        // TODO: explain this in details
        let vs_filepath = concat!(env!("OUT_DIR"), "/shader.vert.spv");
        let fs_filepath = concat!(env!("OUT_DIR"), "/shader.frag.spv");
        let shader = Shader::new(device.clone(), vs_filepath, fs_filepath)?;

        let sub_pass = Subpass::from(render_pass, 0);
        let sub_pass = if sub_pass.is_some() {
            sub_pass.unwrap()
        } else {
            return Err("Impossible to create subpass from renderpass".to_string());
        };

        let (vert_entry_point, frag_entry_point) = shader.entry_points()?;

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
            .vertex_shader(vert_entry_point, ())
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
            .fragment_shader(frag_entry_point, ())
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
                Vertex { position: [-1.0, -1.0], color: [1.0, 0.0, 0.0] },
                Vertex { position: [ 1.0, -1.0], color: [0.0, 1.0, 0.0] },
                Vertex { position: [ 1.0,  1.0], color: [0.0, 0.0, 1.0] },
                Vertex { position: [-1.0, -1.0], color: [1.0, 0.0, 0.0] },
                Vertex { position: [ 1.0,  1.0], color: [0.0, 0.0, 1.0] },
                Vertex { position: [-1.0,  1.0], color: [0.0, 1.0, 0.0] },
            ].iter().cloned()
        ).map_err(|e| format!("Failed to create Vertex Buffers: {}", e))
    }

    fn load_and_create_texture_buffer(queue: &Arc<Queue>)
                                      -> Result<(Arc<ImmutableImage<vulkano::format::R8G8B8A8Srgb>>, CommandBufferExecFuture<NowFuture, AutoCommandBuffer>), String>
    {
        use vulkano::image::Dimensions;
        use image;

        let image = image::load_from_memory_with_format(include_bytes!("../resources/GroundForest003_1k/GroundForest003_COL_VAR1_1K.jpg"),
                                                        image::ImageFormat::JPEG).map_err(|e| format!("Error loading image: {}", e))?.to_rgba();
        let width = image.width();
        let height = image.height();
        let image_data = image.into_raw().clone();

        ImmutableImage::from_iter(
            image_data.iter().cloned(),
            Dimensions::Dim2d { width: width, height: height },
            vulkano::format::R8G8B8A8Srgb,
            queue.clone()).map_err(|e| format!("Error creating the image: {}", e))
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

    fn create_command_buffers(device: &Arc<Device>,
                              graphic_queue: &Arc<Queue>,
                              pipeline: &Arc<GraphicsPipelineAbstract + Send + Sync>,
                              vertex_buffer: &Arc<CpuAccessibleBuffer<[Vertex]>>,
                              texture: &Arc<ImmutableImage<vulkano::format::R8G8B8A8Srgb>>,
                              framebuffers: &Vec<Arc<Framebuffer<Arc<RenderPassAbstract + Send + Sync>, ((), Arc<SwapchainImage>)>>>)
        -> Result<Vec<Arc<AutoCommandBuffer<StandardCommandPoolAlloc>>>, String>
    {
        let sampler = vulkano::sampler::Sampler::new(device.clone(), vulkano::sampler::Filter::Linear,
                                                     vulkano::sampler::Filter::Linear, vulkano::sampler::MipmapMode::Nearest,
                                                     vulkano::sampler::SamplerAddressMode::ClampToEdge,
                                                     vulkano::sampler::SamplerAddressMode::ClampToEdge,
                                                     vulkano::sampler::SamplerAddressMode::ClampToEdge,
                                                     0.0, 1.0, 0.0, 0.0).unwrap();

        use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

        let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_sampled_image(texture.clone(), sampler).unwrap()
            .build().unwrap()
        );

        Ok(framebuffers.iter()
            .map(|framebuffer|
                Arc::new(AutoCommandBufferBuilder::new(
                    device.clone(),
                    graphic_queue.family(),
                ).unwrap()
                .begin_render_pass(
                    framebuffer.clone(),
                    false,
                    vec![[0.0, 0.0, 0.0, 1.0].into(), 1.0.into()],
                ).unwrap()
                .draw(
                    pipeline.clone(),
                    DynamicState::none(),
                    vec![vertex_buffer.clone()],
                    set.clone(),
                    (),
                ).unwrap()
                .end_render_pass().unwrap()
                .build().unwrap())
            ).collect())
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

        acquire_future
            .then_execute(self.vk_graphic_queue.clone(), self.command_buffers[image_num].clone()).unwrap()
            .then_swapchain_present(self.vk_graphic_queue.clone(), self.vk_swapchain.clone(), image_num)
            .then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();
    }
}
