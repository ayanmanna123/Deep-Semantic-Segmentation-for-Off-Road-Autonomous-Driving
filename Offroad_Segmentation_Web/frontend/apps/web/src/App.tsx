import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@workspace/ui/components/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@workspace/ui/components/tabs"
import { 
  TrendingUp, 
  BarChart3, 
  PieChart, 
  Grid, 
  Cpu, 
  Layers, 
  Video, 
  Target, 
  Map, 
  Sliders
} from "lucide-react"

const CLASS_COLORS: Record<string, string> = {
  "Trees": "#228B22",
  "Lush Bushes": "#00FF00",
  "Dry Grass": "#D2B48C",
  "Dry Bushes": "#8B5A2B",
  "Ground Clutter": "#808000",
  "Flowers": "#FF69B4",
  "Logs": "#8B4513",
  "Rocks": "#808080",
  "Landscape": "#A0522D",
  "Sky": "#87CEEB",
}

const CLASS_DESCRIPTIONS: Record<string, string> = {
  "Trees": "Dense forestation and high foliage blocking offroad travel.",
  "Lush Bushes": "Dense living green bushes requiring navigation detours.",
  "Dry Grass": "Dry, low-lying grass fields providing a relatively safe driving surface.",
  "Dry Bushes": "Dead or dry shrubbery posing low-to-medium clearance risk.",
  "Ground Clutter": "Fallen branches, twigs, and loose organic debris on the ground.",
  "Flowers": "Wildflower patches to be identified and preserved where possible.",
  "Logs": "Heavy wood logs posing severe collision hazards.",
  "Rocks": "Solid stone blockages requiring immediate obstacle avoidance.",
  "Landscape": "Soil, sand, and generic traversable desert terrain.",
  "Sky": "Clear skyline used for visual orientation and pitch adjustment.",
}

export function App() {
  return (
    <div className="min-h-screen bg-[#0a0a0c] p-8 text-slate-100 font-sans selection:bg-indigo-500/30">
      <div className="mx-auto max-w-7xl space-y-8">
        <header className="space-y-2">
          <h1 className="text-5xl font-extrabold tracking-tighter text-white mt-4">
            Offroad <span className="text-indigo-400 italic">Segmentation</span> Explorer
          </h1>
          <p className="text-xl text-slate-400 font-light">Precision analytics for rugged terrain perception.</p>
        </header>

        {/* Highlight Stats Dashboard */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card className="border border-white/5 bg-slate-900/30 backdrop-blur-xl ring-1 ring-white/5 shadow-xl">
            <CardContent className="p-6 flex flex-col gap-4">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Model mIoU</span>
                <span className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400">
                  <Target className="h-5 w-5" />
                </span>
              </div>
              <div>
                <h3 className="text-3xl font-extrabold text-white">0.5029</h3>
                <p className="text-[10px] text-indigo-300 font-medium mt-1">DeepLabV3+ EfficientNet-B5</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border border-white/5 bg-slate-900/30 backdrop-blur-xl ring-1 ring-white/5 shadow-xl">
            <CardContent className="p-6 flex flex-col gap-4">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Dataset Size</span>
                <span className="p-2 rounded-xl bg-purple-500/10 text-purple-400">
                  <Layers className="h-5 w-5" />
                </span>
              </div>
              <div>
                <h3 className="text-3xl font-extrabold text-white">10,000+</h3>
                <p className="text-[10px] text-purple-300 font-medium mt-1">High-Res Annotated Dataset</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border border-white/5 bg-slate-900/30 backdrop-blur-xl ring-1 ring-white/5 shadow-xl">
            <CardContent className="p-6 flex flex-col gap-4">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Architecture</span>
                <span className="p-2 rounded-xl bg-emerald-500/10 text-emerald-400">
                  <Cpu className="h-5 w-5" />
                </span>
              </div>
              <div>
                <h3 className="text-2xl font-extrabold text-white">DeepLabV3+</h3>
                <p className="text-[10px] text-emerald-300 font-medium mt-1">SMP + PyTorch Engine</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border border-white/5 bg-slate-900/30 backdrop-blur-xl ring-1 ring-white/5 shadow-xl">
            <CardContent className="p-6 flex flex-col gap-4">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Loss Functions</span>
                <span className="p-2 rounded-xl bg-amber-500/10 text-amber-400">
                  <Sliders className="h-5 w-5" />
                </span>
              </div>
              <div>
                <h3 className="text-xl font-extrabold text-white leading-snug">Dice + Focal + Lovasz</h3>
                <p className="text-[10px] text-amber-300 font-medium mt-1">Optimized Class Balancing</p>
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full max-w-3xl grid-cols-5 p-1 bg-slate-900/60 backdrop-blur-sm rounded-xl border border-white/5">
            <TabsTrigger value="overview" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">Overview</TabsTrigger>
            <TabsTrigger value="metrics" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">Training Metrics</TabsTrigger>
            <TabsTrigger value="iou" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">Class IoU</TabsTrigger>
            <TabsTrigger value="matrix" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">Confusion Matrix</TabsTrigger>
            <TabsTrigger value="dist" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">Dataset</TabsTrigger>
          </TabsList>

          {/* Overview Tab Content */}
          <TabsContent value="overview" className="mt-8 space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2 space-y-6">
                {/* About, Impact & Use Cases Card */}
                <Card className="border border-white/5 shadow-xl bg-slate-900/40 backdrop-blur-xl rounded-2xl ring-1 ring-white/5">
                  <CardHeader className="bg-white/[0.02] border-b border-white/5 flex flex-row items-center gap-4 p-8">
                    <Map className="h-7 w-7 text-indigo-400 shrink-0" />
                    <div>
                      <CardTitle className="text-3xl font-extrabold text-white tracking-tight">Project Overview, Impact & Use Cases</CardTitle>
                      <CardDescription className="text-base text-slate-400 mt-1">Understanding offroad perception engineering, its real-world impact, and deployment scenarios.</CardDescription>
                    </div>
                  </CardHeader>
                  <CardContent className="p-8 space-y-8">
                    {/* The Project Section */}
                    <div className="space-y-4">
                      <h3 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
                        <span className="h-4 w-1.5 bg-indigo-500 rounded-full"></span>
                        The Project
                      </h3>
                      <div className="space-y-4 text-base lg:text-lg leading-relaxed text-slate-300">
                        <p>
                          Standard autonomous vehicles navigate using structured roads, painted lane markings, and uniform surroundings. In offroad environments, these reference points are completely absent. Autonomous systems must cross open deserts, thick shrubbery, rocks, and logs.
                        </p>
                        <p>
                          This project implements a state-of-the-art semantic scene segmentation algorithm built specifically for offroad navigation. Powered by a <strong className="text-indigo-300 font-semibold">DeepLabV3+</strong> architecture and a high-capacity <strong className="text-indigo-300 font-semibold">EfficientNet-B5</strong> encoder, our model segments complex textures in real-time, allowing automated path planners to identify safe paths through rugged terrains.
                        </p>
                      </div>
                    </div>

                    {/* Impact & Use Cases Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 pt-8 border-t border-white/5">
                      {/* Real-World Impact Section */}
                      <div className="space-y-4">
                        <h3 className="text-xl lg:text-2xl font-bold text-indigo-400 tracking-tight flex items-center gap-2">
                          <span className="h-4 w-1.5 bg-indigo-500 rounded-full"></span>
                          Real-World Impact
                        </h3>
                        <div className="space-y-4">
                          <div className="border-l-4 border-indigo-500 bg-gradient-to-r from-indigo-500/10 to-transparent p-4 rounded-r-xl space-y-1 transition-all hover:from-indigo-500/15 duration-300">
                            <h4 className="text-sm lg:text-base font-bold text-white">Disaster & Crisis Navigation</h4>
                            <p className="text-xs lg:text-sm text-slate-400 leading-relaxed mt-1">
                              In extreme events like earthquakes or urban conflict zones where infrastructure is destroyed and buildings have collapsed, our model identifies the safest paths to successfully navigate the debris and reach the destination.
                            </p>
                          </div>
                          
                          <div className="border-l-4 border-indigo-500 bg-gradient-to-r from-indigo-500/10 to-transparent p-4 rounded-r-xl space-y-1 transition-all hover:from-indigo-500/15 duration-300">
                            <h4 className="text-sm lg:text-base font-bold text-white">Collision Avoidance</h4>
                            <p className="text-xs lg:text-sm text-slate-400 leading-relaxed mt-1">
                              Prevents structural chassis damage by identifying rocks and logs that traditional sensors might miss.
                            </p>
                          </div>

                          <div className="border-l-4 border-indigo-500 bg-gradient-to-r from-indigo-500/10 to-transparent p-4 rounded-r-xl space-y-1 transition-all hover:from-indigo-500/15 duration-300">
                            <h4 className="text-sm lg:text-base font-bold text-white">High-Speed Traversability</h4>
                            <p className="text-xs lg:text-sm text-slate-400 leading-relaxed mt-1">
                              Promotes real-time path planning, allowing vehicles to maintain speed rather than crawling over scanned terrain.
                            </p>
                          </div>

                          <div className="border-l-4 border-indigo-500 bg-gradient-to-r from-indigo-500/10 to-transparent p-4 rounded-r-xl space-y-1 transition-all hover:from-indigo-500/15 duration-300">
                            <h4 className="text-sm lg:text-base font-bold text-white">Robust Generalization</h4>
                            <p className="text-xs lg:text-sm text-slate-400 leading-relaxed mt-1">
                              Operates under challenging offroad conditions such as dynamic lighting, harsh shadows, and dusty environments.
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Key Use Cases Section */}
                      <div className="space-y-4">
                        <h3 className="text-xl lg:text-2xl font-bold text-purple-400 tracking-tight flex items-center gap-2">
                          <span className="h-4 w-1.5 bg-purple-500 rounded-full"></span>
                          Key Use Cases
                        </h3>
                        <div className="space-y-4">
                          <div className="border-l-4 border-purple-500 bg-gradient-to-r from-purple-500/10 to-transparent p-4 rounded-r-xl space-y-1 transition-all hover:from-purple-500/15 duration-300">
                            <h4 className="text-sm lg:text-base font-bold text-white">Tactical Defense (UGVs)</h4>
                            <p className="text-xs lg:text-sm text-slate-400 leading-relaxed mt-1">
                              Deploying unmanned ground vehicles for reconnaissance without human personnel exposure.
                            </p>
                          </div>

                          <div className="border-l-4 border-purple-500 bg-gradient-to-r from-purple-500/10 to-transparent p-4 rounded-r-xl space-y-1 transition-all hover:from-purple-500/15 duration-300">
                            <h4 className="text-sm lg:text-base font-bold text-white">Search & Rescue (SAR)</h4>
                            <p className="text-xs lg:text-sm text-slate-400 leading-relaxed mt-1">
                              Navigating remote, disaster-stricken areas where traditional roads have been destroyed.
                            </p>
                          </div>

                          <div className="border-l-4 border-purple-500 bg-gradient-to-r from-purple-500/10 to-transparent p-4 rounded-r-xl space-y-1 transition-all hover:from-purple-500/15 duration-300">
                            <h4 className="text-sm lg:text-base font-bold text-white">Planetary Exploration</h4>
                            <p className="text-xs lg:text-sm text-slate-400 leading-relaxed mt-1">
                              Autonomously steering exploration rovers through rocky ground clutter and sandy craters.
                            </p>
                          </div>

                          <div className="border-l-4 border-purple-500 bg-gradient-to-r from-purple-500/10 to-transparent p-4 rounded-r-xl space-y-1 transition-all hover:from-purple-500/15 duration-300">
                            <h4 className="text-sm lg:text-base font-bold text-white">Industrial Automation</h4>
                            <p className="text-xs lg:text-sm text-slate-400 leading-relaxed mt-1">
                              Operating large-scale automated machinery in remote mining, agriculture, and forestry sites.
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Video Demonstration Card */}
                <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl rounded-2xl overflow-hidden ring-1 ring-white/5">
                  <CardHeader className="bg-white/[0.02] border-b border-white/5 flex flex-row items-center gap-3">
                    <Video className="h-6 w-6 text-indigo-400" />
                    <div>
                      <CardTitle className="text-white">Offroad Navigation Demonstration</CardTitle>
                      <CardDescription className="text-slate-400">
                        Real-time model prediction and A* path planning overlay on unstructured desert terrain.
                      </CardDescription>
                    </div>
                  </CardHeader>
                  <CardContent className="p-6 bg-black flex justify-center">
                    <div className="w-full max-w-4xl rounded-xl overflow-hidden shadow-inner border border-white/10">
                      <video 
                        src="https://ik.imagekit.io/h0gbowxm32/offroad_path_demonstration_realtime.mp4?updatedAt=1777880209651" 
                        controls 
                        autoPlay 
                        loop 
                        muted 
                        className="w-full object-cover"
                      />
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Sidebar Guide */}
              <div className="space-y-6">
                <Card className="border border-white/5 shadow-xl bg-slate-900/40 backdrop-blur-xl rounded-2xl ring-1 ring-white/5">
                  <CardHeader className="bg-white/[0.02] border-b border-white/5">
                    <CardTitle className="text-white flex items-center gap-2">
                      <Grid className="h-5 w-5 text-indigo-400" />
                      Terrain Class Legend
                    </CardTitle>
                    <CardDescription className="text-slate-400">
                      Color codes mapping model outputs to terrain categories.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-6">
                    <div className="space-y-4">
                      {Object.entries(CLASS_COLORS).map(([cls, color]) => (
                        <div key={cls} className="flex gap-4 items-start border-b border-white/[0.03] pb-3 last:border-0 last:pb-0">
                          <span 
                            className="h-5 w-5 rounded-full shrink-0 shadow-lg border border-white/10" 
                            style={{ backgroundColor: color }}
                          />
                          <div className="space-y-0.5">
                            <h4 className="text-xs font-bold text-white uppercase tracking-wider">{cls}</h4>
                            <p className="text-[11px] text-slate-400 font-light leading-normal">
                              {CLASS_DESCRIPTIONS[cls]}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Metrics Tab Content */}
          <TabsContent value="metrics" className="mt-8">
            <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl rounded-2xl overflow-hidden ring-1 ring-white/5">
              <CardHeader className="bg-white/[0.02] border-b border-white/5 flex flex-row items-center gap-3">
                <TrendingUp className="h-6 w-6 text-indigo-400" />
                <div>
                  <CardTitle className="text-white">Training Metrics Analysis</CardTitle>
                  <CardDescription className="text-slate-400">
                    Loss minimization and Intersection over Union (IoU) curves across epochs.
                  </CardDescription>
                </div>
              </CardHeader>
              <CardContent className="p-12 space-y-8 flex flex-col items-center">
                <div className="p-4 bg-white/[0.02] rounded-2xl shadow-2xl border border-white/5 backdrop-blur-md max-w-3xl">
                  <img src="/metrics_plot.png" alt="Training Metrics" className="max-h-[600px] rounded-lg" />
                </div>
                <div className="max-w-3xl text-sm text-slate-300 leading-relaxed space-y-2 text-center">
                  <p className="font-bold text-white">Loss & Accuracy Convergence</p>
                  <p>
                    The training graphs demonstrate steady optimization of the hybrid <strong>Dice + Focal + Lovasz loss</strong>. 
                    The validation IoU converges nicely toward our peak performance metrics, showing strong generalization capabilities with zero signs of severe overfitting due to robust data augmentation.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Class IoU Tab Content */}
          <TabsContent value="iou" className="mt-8">
            <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl rounded-2xl overflow-hidden ring-1 ring-white/5">
              <CardHeader className="bg-white/[0.02] border-b border-white/5 flex flex-row items-center gap-3">
                <Target className="h-6 w-6 text-indigo-400" />
                <div>
                  <CardTitle className="text-white">Per-Class Intersection over Union</CardTitle>
                  <CardDescription className="text-slate-400">
                    Accuracy validation across individual terrain classes.
                  </CardDescription>
                </div>
              </CardHeader>
              <CardContent className="p-12 space-y-8 flex flex-col items-center">
                <div className="p-4 bg-white/[0.02] rounded-2xl shadow-2xl border border-white/5 backdrop-blur-md max-w-3xl">
                  <img src="/per_class_iou.png" alt="Per-Class IoU" className="max-h-[600px] rounded-lg invert brightness-110 contrast-125 hue-rotate-180" />
                </div>
                <div className="max-w-3xl text-sm text-slate-300 leading-relaxed space-y-2 text-center">
                  <p className="font-bold text-white">Understanding Per-Class Performance</p>
                  <p>
                    Classes like <strong>Sky</strong> and <strong>Landscape</strong> achieve extremely high IoU values (above 0.8), 
                    since they cover wide continuous regions. Unstructured classes like <strong>Rocks</strong> and <strong>Logs</strong> are highly sporadic and represent significant challenges due to size variety, but the model maintains a strong detection threshold, essential for avoid-guidance.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Confusion Matrix Tab Content */}
          <TabsContent value="matrix" className="mt-8">
            <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl rounded-2xl overflow-hidden ring-1 ring-white/5">
              <CardHeader className="bg-white/[0.02] border-b border-white/5 flex flex-row items-center gap-3">
                <Grid className="h-6 w-6 text-indigo-400" />
                <div>
                  <CardTitle className="text-white">Confusion Matrix</CardTitle>
                  <CardDescription className="text-slate-400">
                    Detailed breakdown of target class predictions vs ground truth.
                  </CardDescription>
                </div>
              </CardHeader>
              <CardContent className="p-12 space-y-8 flex flex-col items-center">
                <div className="p-4 bg-white/[0.02] rounded-2xl shadow-2xl border border-white/5 backdrop-blur-md max-w-3xl">
                  <img src="/confusion_matrix.png" alt="Confusion Matrix" className="max-h-[600px] rounded-lg invert brightness-110 contrast-125 hue-rotate-180" />
                </div>
                <div className="max-w-3xl text-sm text-slate-300 leading-relaxed space-y-2 text-center">
                  <p className="font-bold text-white">Prediction Error Analysis</p>
                  <p>
                    The confusion matrix helps identify visual overlap. For example, the model occasionally confuses 
                    <strong>Dry Grass</strong> with <strong>Dry Bushes</strong>, which is expected due to their highly similar 
                    color profiles and visual textures in dry offroad settings. This matrix guided our augmentation strategy to emphasize texture differentiation.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Dataset Distribution Tab Content */}
          <TabsContent value="dist" className="mt-8">
            <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl rounded-2xl overflow-hidden ring-1 ring-white/5">
              <CardHeader className="bg-white/[0.02] border-b border-white/5 flex flex-row items-center gap-3">
                <PieChart className="h-6 w-6 text-indigo-400" />
                <div>
                  <CardTitle className="text-white">Dataset Distribution</CardTitle>
                  <CardDescription className="text-slate-400">
                    Represented pixel ratio and class balance in the dataset.
                  </CardDescription>
                </div>
              </CardHeader>
              <CardContent className="p-12 space-y-8 flex flex-col items-center">
                <div className="p-4 bg-white/[0.02] rounded-2xl shadow-2xl border border-white/5 backdrop-blur-md max-w-3xl">
                  <img src="/class_distribution.png" alt="Class Distribution" className="max-h-[600px] rounded-lg invert brightness-110 contrast-125 hue-rotate-180" />
                </div>
                <div className="max-w-3xl text-sm text-slate-300 leading-relaxed space-y-2 text-center">
                  <p className="font-bold text-white">Tackling Imbalance with Lovasz & Focal Loss</p>
                  <p>
                    Offroad scene datasets suffer from massive imbalance: <strong>Landscape</strong> and <strong>Sky</strong> comprise 
                    the vast majority of pixels, while high-danger hazards like <strong>Logs</strong> and <strong>Rocks</strong> cover a minimal fraction. 
                    Using focal loss and Lovasz loss allows our DeepLabV3+ head to prioritize these sparse classes, ensuring safe autonomous navigation.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <footer className="text-center text-sm text-slate-500 py-20 border-t border-white/5">
          <div className="flex justify-center gap-8 mb-4">
            <span className="opacity-40">Duality AI Challenge</span>
            <span className="opacity-40">Offroad Semantic Segmentation</span>
            <span className="opacity-40">© 2026</span>
          </div>
          <p className="font-light tracking-widest uppercase text-[10px] text-slate-600">Built for high-performance offroad perception</p>
        </footer>
      </div>
    </div>
  )
}
