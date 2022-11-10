##Thanks for graph-cut-ransac-master (CVPR2020)

py::tuple findHomography_(
	py::array_t<double> correspondeces_,
	int h1,int w1 ,int h2 ,int w2
)

int findHomography_(

	neighborhood::NeighborhoodGraph

	utils::DefaultHomographyEstimator 

	sampler::Sampler

	sampler::UniformSampler

	GCRANSAC<utils::DefaultHomographyEstimator,
					AbstractNeighborhood,
					MSACScoringFunction<utils::DefaultHomographyEstimator>,
					preemption::SPRTPreemptiveVerfication<utils::DefaultHomographyEstimator>,
					inlier_selector::SpacePartitioningRANSAC<utils::DefaultHomographyEstimator, AbstractNeighborhood>> gcransac;
					
	gcransac.run(

	delete neighborhood_graph_ptr;

