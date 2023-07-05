package com.google.inject.assistedinject.subpkg;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;
import static org.junit.Assume.assumeTrue;
def load_sd_from_config(ckpt, verbose=False):
    print(f”Loading model from {ckpt}“)
    pl_sd = torch.load(ckpt, map_location=“cuda”)
    if “global_step” in pl_sd:
        print(f”Global Step: {pl_sd[‘global_step’]}“)
    sd = pl_sd[“state_dict”]
    return sd
def crash(e, s):
    global model
    global device
    print(s, ‘\n’, e)
    del model
    del device
    print(‘exiting...calling os._exit(0)‘)
    t = threading.Timer(0.25, os._exit, args=[0])
    t.start()
class MemUsageMonitor(threading.Thread):
    stop_flag = False
    max_usage = 0
    total = -1
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
    def run(self):
        try:
            pynvml.nvmlInit()
        except:
            print(f”[{self.name}] Unable to initialize NVIDIA management. No memory stats. \n”)
            return
        print(f”[{self.name}] Recording max memory usage...\n”)
        handle = pynvml.nvmlDeviceGetHandleByIndex(opt.gpu)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        while not self.stop_flag:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.max_usage = max(self.max_usage, m.used)
            # print(self.max_usage)
            time.sleep(0.1)
        print(f”[{self.name}] Stopped recording.\n”)
        pynvml.nvmlShutdown()
    def read(self):
        return self.max_usage, self.total
    def stop(self):
        self.stop_flag = True
    def read_and_stop(self):
        self.stop_flag = True
        return self.max_usage, self.total
class CFGMaskedDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
        x_in = x
        x_in = torch.cat([x_in] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        denoised = uncond + (cond - uncond) * cond_scale
        if mask is not None:
            assert x0 is not None
            img_orig = x0
            mask_inv = 1. - mask
            denoised = (img_orig * mask_inv) + (mask * denoised)
        return denoised
class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale
class KDiffusionSampler:
    def __init__(self, m, sampler):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler
    def get_sampler_name(self):
        return self.schedule
    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T):
        sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)
        samples_ddim = K.sampling.__dict__[f’sample_{self.schedule}’](model_wrap_cfg, x, sigmas, extra_args={‘cond’: conditioning, ‘uncond’: unconditional_conditioning, ‘cond_scale’: unconditional_guidance_scale}, disable=False)
        return samples_ddim, None
def create_random_tensors(shape, seeds):
    xs = []
    for seed in seeds:
        torch.manual_seed(seed)
        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it’s better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this so i do not dare change it for now because
        # it will break everyone’s seeds.
        xs.append(torch.randn(shape, device=device))
    x = torch.stack(xs)
    return x
def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
def load_GFPGAN():
    model_name = ‘GFPGANv1.3’
    model_path = os.path.join(GFPGAN_dir, ‘experiments/pretrained_models’, model_name + ‘.pth’)
    if not os.path.isfile(model_path):
        raise Exception(“GFPGAN model not found at path “+model_path)
    sys.path.append(os.path.abspath(GFPGAN_dir))
    from gfpgan import GFPGANer
    instance = GFPGANer(model_path=model_path, upscale=1, arch=‘clean’, channel_multiplier=2, bg_upsampler=None)
    if opt.gfpgan_cpu or opt.extra_models_cpu:
        instance.device = torch.device(‘cpu’)
    else:
        instance.device = torch.device(f’cuda:{opt.gpu}’) # another way to set gpu device
    return instance
def load_RealESRGAN(model_name: str):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    RealESRGAN_models = {
        ‘RealESRGAN_x2plus’: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
        ‘RealESRGAN_x4plus_anime_6B’: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    }
    model_path = os.path.join(RealESRGAN_dir, ‘experiments/pretrained_models’, model_name + ‘.pth’)
    if not os.path.isfile(model_path):
        raise Exception(model_name+“.pth not found at path “+model_path)
    sys.path.append(os.path.abspath(RealESRGAN_dir))
    from realesrgan import RealESRGANer
    if opt.esrgan_cpu or opt.extra_models_cpu:
        instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=False)
        instance.model.name = model_name
        instance.device = torch.device(‘cpu’)
        instance.device = torch.device(‘cpu’)
        instance.model.to(‘cpu’)
    else:
        instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=not opt.no_half)
        instance.model.name = model_name
        instance.device = torch.device(f’cuda:{opt.gpu}’) # another way to set gpu device
    return instance

import com.google.common.base.StandardSystemProperty;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.inject.AbstractModule;
import com.google.inject.CreationException;
import com.google.inject.Guice;
import com.google.inject.Injector;
import com.google.inject.Key;
import com.google.inject.assistedinject.Assisted;
import com.google.inject.assistedinject.AssistedInject;
import com.google.inject.assistedinject.FactoryModuleBuilder;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import javax.inject.Inject;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that run in a subpackage, to make sure tests aren't passing because they're run in the same
 * package as the assistedinject code.
 *
 * <p>See https://github.com/google/guice/issues/904
 */
@RunWith(JUnit4.class)
public final class SubpackageTest {
  private static final double JAVA_VERSION =
      Double.parseDouble(StandardSystemProperty.JAVA_SPECIFICATION_VERSION.value());

  private static final MethodHandles.Lookup LOOKUPS = MethodHandles.lookup();

  private final Logger loggerToWatch = Logger.getLogger(AssistedInject.class.getName());

  private final List<LogRecord> logRecords = Lists.newArrayList();
  private final Handler fakeHandler =
      new Handler() {
        @Override
        public void publish(LogRecord logRecord) {
          logRecords.add(logRecord);
        }

        @Override
        public void flush() {}

        @Override
        public void close() {}
      };

  @Before
  public void setUp() throws Exception {
    loggerToWatch.addHandler(fakeHandler);
    setAllowPrivateLookupFallback(true);
    setAllowMethodHandleWorkaround(true);
  }

  @After
  public void tearDown() throws Exception {
    loggerToWatch.removeHandler(fakeHandler);
    setAllowPrivateLookupFallback(true);
    setAllowMethodHandleWorkaround(true);
  }

  public abstract static class AbstractAssisted {
    interface Factory<O extends AbstractAssisted, I extends CharSequence> {
      O create(I string);
    }
  }

  static class ConcreteAssisted extends AbstractAssisted {
    @Inject
    ConcreteAssisted(@SuppressWarnings("unused") @Assisted String string) {}
  }

  static class ConcreteAssistedWithOverride extends AbstractAssisted {
    @AssistedInject
    ConcreteAssistedWithOverride(@SuppressWarnings("unused") @Assisted String string) {}

    @AssistedInject
    ConcreteAssistedWithOverride(@SuppressWarnings("unused") @Assisted StringBuilder sb) {}

    interface Factory extends AbstractAssisted.Factory<ConcreteAssistedWithOverride, String> {
      @Override
      ConcreteAssistedWithOverride create(String string);
    }

    interface Factory2 extends AbstractAssisted.Factory<ConcreteAssistedWithOverride, String> {
      @Override
      ConcreteAssistedWithOverride create(String string);

      ConcreteAssistedWithOverride create(StringBuilder sb);
    }
  }

  static class ConcreteAssistedWithoutOverride extends AbstractAssisted {
    @Inject
    ConcreteAssistedWithoutOverride(@SuppressWarnings("unused") @Assisted String string) {}

    interface Factory extends AbstractAssisted.Factory<ConcreteAssistedWithoutOverride, String> {}
  }

  public static class Public extends AbstractAssisted {
    @AssistedInject
    Public(@SuppressWarnings("unused") @Assisted String string) {}

    @AssistedInject
    Public(@SuppressWarnings("unused") @Assisted StringBuilder sb) {}

    public interface Factory extends AbstractAssisted.Factory<Public, String> {
      @Override
      Public create(String string);

      Public create(StringBuilder sb);
    }
  }

  @Test
  public void testNoPrivateFallbackOrWorkaround() throws Exception {
    setAllowMethodHandleWorkaround(false);
    setAllowPrivateLookupFallback(false);

    if (JAVA_VERSION > 1.8) {
      // Above 1.8 will fail, because they can't access private details w/o the workarounds.
      try {
        Guice.createInjector(
            new AbstractModule() {
              @Override
              protected void configure() {
                install(
                    new FactoryModuleBuilder().build(ConcreteAssistedWithOverride.Factory.class));
              }
            });
        fail("Expected CreationException");
      } catch (CreationException ce) {
        assertThat(Iterables.getOnlyElement(ce.getErrorMessages()).getMessage())
            .contains("Please call FactoryModuleBuilder.withLookups");
      }
      LogRecord record = Iterables.getOnlyElement(logRecords);
      assertThat(record.getMessage()).contains("Please pass a `MethodHandles.lookup()`");
    } else {
      // 1.8 & below will succeed, because that's the only way they can work.
      Injector injector =
          Guice.createInjector(
              new AbstractModule() {
                @Override
                protected void configure() {
                  install(
                      new FactoryModuleBuilder().build(ConcreteAssistedWithOverride.Factory.class));
                }
              });
      LogRecord record = Iterables.getOnlyElement(logRecords);
      assertThat(record.getMessage()).contains("Please pass a `MethodHandles.lookup()`");

      ConcreteAssistedWithOverride.Factory factory =
          injector.getInstance(ConcreteAssistedWithOverride.Factory.class);
      factory.create("foo");
      AbstractAssisted.Factory<ConcreteAssistedWithOverride, String> factoryAbstract = factory;
      factoryAbstract.create("foo");
    }
  }

  @Test
  public void testHandleWorkaroundOnly() throws Exception {
    setAllowPrivateLookupFallback(false);

    Injector injector =
        Guice.createInjector(
            new AbstractModule() {
              @Override
              protected void configure() {
                install(
                    new FactoryModuleBuilder().build(ConcreteAssistedWithOverride.Factory.class));
              }
            });
    LogRecord record = Iterables.getOnlyElement(logRecords);
    assertThat(record.getMessage()).contains("Please pass a `MethodHandles.lookup()`");

    ConcreteAssistedWithOverride.Factory factory =
        injector.getInstance(ConcreteAssistedWithOverride.Factory.class);
    factory.create("foo");
    AbstractAssisted.Factory<ConcreteAssistedWithOverride, String> factoryAbstract = factory;
    factoryAbstract.create("foo");
  }

  @Test
  public void testGeneratedDefaultMethodsForwardCorrectly() throws Exception {
    // This test requires above java 1.8.
    // 1.8's reflection capability is tested via "testReflectionFallbackWorks".
    assumeTrue(JAVA_VERSION > 1.8);

    final Key<AbstractAssisted.Factory<ConcreteAssisted, String>> concreteKey =
        new Key<AbstractAssisted.Factory<ConcreteAssisted, String>>() {};
    Injector injector =
        Guice.createInjector(
            new AbstractModule() {
              @Override
              protected void configure() {
                install(
                    new FactoryModuleBuilder()
                        .withLookups(LOOKUPS)
                        .build(ConcreteAssistedWithOverride.Factory.class));
                install(
                    new FactoryModuleBuilder()
                        .withLookups(LOOKUPS)
                        .build(ConcreteAssistedWithOverride.Factory2.class));
                install(
                    new FactoryModuleBuilder()
                        .build(ConcreteAssistedWithoutOverride.Factory.class));
                install(new FactoryModuleBuilder().build(Public.Factory.class));
                install(new FactoryModuleBuilder().build(concreteKey));
              }
            });
    assertThat(logRecords).isEmpty();

    ConcreteAssistedWithOverride.Factory factory1 =
        injector.getInstance(ConcreteAssistedWithOverride.Factory.class);
    factory1.create("foo");
    AbstractAssisted.Factory<ConcreteAssistedWithOverride, String> factory1Abstract = factory1;
    factory1Abstract.create("foo");

    ConcreteAssistedWithOverride.Factory2 factory2 =
        injector.getInstance(ConcreteAssistedWithOverride.Factory2.class);
    factory2.create("foo");
    factory2.create(new StringBuilder("foo"));
    AbstractAssisted.Factory<ConcreteAssistedWithOverride, String> factory2Abstract = factory2;
    factory2Abstract.create("foo");

    ConcreteAssistedWithoutOverride.Factory factory3 =
        injector.getInstance(ConcreteAssistedWithoutOverride.Factory.class);
    factory3.create("foo");
    AbstractAssisted.Factory<ConcreteAssistedWithoutOverride, String> factory3Abstract = factory3;
    factory3Abstract.create("foo");

    Public.Factory factory4 = injector.getInstance(Public.Factory.class);
    factory4.create("foo");
    factory4.create(new StringBuilder("foo"));
    AbstractAssisted.Factory<Public, String> factory4Abstract = factory4;
    factory4Abstract.create("foo");

    AbstractAssisted.Factory<ConcreteAssisted, String> factory5 = injector.getInstance(concreteKey);
    factory5.create("foo");
  }

  private static void setAllowPrivateLookupFallback(boolean allowed) throws Exception {
    Class<?> factoryProvider2 = Class.forName("com.google.inject.assistedinject.FactoryProvider2");
    Field field = factoryProvider2.getDeclaredField("allowPrivateLookupFallback");
    field.setAccessible(true);
    field.setBoolean(null, allowed);
  }

  private static void setAllowMethodHandleWorkaround(boolean allowed) throws Exception {
    Class<?> factoryProvider2 = Class.forName("com.google.inject.assistedinject.FactoryProvider2");
    Field field = factoryProvider2.getDeclaredField("allowMethodHandleWorkaround");
    field.setAccessible(true);
    field.setBoolean(null, allowed);
  }
}
