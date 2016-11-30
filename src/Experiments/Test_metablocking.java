package Experiments;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import BlockBuilding.MemoryBased.AttributeClusteringBlockingEntropy;
import BlockBuilding.MemoryBased.TokenBlocking;
import BlockProcessing.AbstractEfficiencyMethod;
import BlockProcessing.BlockRefinement.BlockFiltering;
import BlockProcessing.BlockRefinement.ComparisonsBasedBlockPurging;
import BlockProcessing.ComparisonRefinement.BilateralDuplicatePropagation;
import DataStructures.AbstractBlock;
import DataStructures.EntityProfile;
import DataStructures.IdDuplicates;
import MetaBlocking.ThresholdWeightingScheme;
import MetaBlocking.WeightingScheme;
import OnTheFlyMethods.FastImplementations.BlastWeightedNodePruning;
import SupervisedMetablocking.SupervisedWEP;
import Utilities.BlockStatistics;
import Utilities.ExecuteBlockComparisons;
import Utilities.RepresentationModel;
import Utilities.SerializationUtilities;
import libsvm.svm_parameter;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
//import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.trees.J48;

/**
 *
 * @author gap2
 */
public class Test_metablocking {

	private final static int ITERATIONS = 10;
	//String  atributos_value[][] = {{ "id", "name" , "age","postcode", "state","given_name","date_of_birth","suburb","address_2","address_1","surname","soc_sec_id","phone_number","street_number"},{ "id", "title" , "authors", "venue", "year"}};
	
//	private static Classifier[] getSupervisedCepClassifiers() {
//		NaiveBayes naiveBayes = new NaiveBayes();
//		naiveBayes.setUseKernelEstimator(false);
//		naiveBayes.setUseSupervisedDiscretization(true);
//
//		BayesNet bayesNet = new BayesNet();
//
//		Classifier[] classifiers = new Classifier[2];
//		classifiers[0] = naiveBayes;
//		classifiers[1] = bayesNet;
//		return classifiers;
//	}

//	private static Classifier[] getSupervisedCnpClassifiers() {
//		return getSupervisedCepClassifiers();
//	}

	private static Classifier[] getSupervisedWepClassifiers() {
		  NaiveBayes naiveBayes = new NaiveBayes();
	        naiveBayes.setUseKernelEstimator(false);
	        naiveBayes.setUseSupervisedDiscretization(false);

	        J48 j48 = new J48();
	        j48.setMinNumObj(5);
	        j48.setConfidenceFactor((float) 0.10);

	        SMO smo = new SMO();
	       //((Object) smo).setBuildLogisticModels(true);
	        smo.setKernel(new PolyKernel());
	        smo.setC(9.0);

	        BayesNet bayesNet = new BayesNet();

	        Classifier[] classifiers = new Classifier[1];
	        classifiers[0] = naiveBayes;
	        //classifiers[1] = smo;
	        //classifiers[2] = smo;
	        //classifiers[3] = bayesNet;
	        return classifiers;
	}


	public static svm_parameter get_grid(svm_parameter param) {
		PrintWriter out =null;
		BufferedReader in=null;
		BufferedReader in_t=null;
		Process proc = null;
		String line=null;		
		String userHome = System.getProperty("user.home");
		System.out.println(userHome);
		try {
			proc = Runtime.getRuntime().exec("/bin/bash", null, new File(userHome+"/Downloads/libsvm-3.21/tools/"));
			if (proc != null) {
				in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
				out= new PrintWriter(new BufferedWriter(new OutputStreamWriter(proc.getOutputStream())), true);
			}
		}
		catch (IOException e) {
			e.printStackTrace();
		}

		if (proc != null) {
			System.out.println("antes do grid.py----------python grid.py /tmp/test.libsvm");
			out.println("python grid.py /tmp/test.libsvm");//reverrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
			try {
				while ((line = in.readLine()) != null ) {
					if(!line.contains("[local]")){
						String arg[]=line.split(" ");
						param.C= Double.parseDouble(arg[0]);
						param.gamma= Double.parseDouble(arg[1]);
						System.out.println("valor c  " + param.C + "  " + param.gamma);
						return param;
					}

				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		System.out.println("depois do grid.py----------");
		System.out.println("depois do grid.py----------");
		return null;    
	}

	

//	private static ArrayList<Comparison>[] testeParaGerarAscomparações(List<AbstractBlock> blocks, int[] nblocks, ExecuteBlockComparisons ebc) {
//
//		List<AbstractBlock> blocks_select = new ArrayList<AbstractBlock>();
//		ArrayList<Comparison>[] listComparison = (ArrayList<Comparison>[])new ArrayList[10];
//		EntityIndex entityIndex = new EntityIndex(blocks);
//		for (int i = 0; i < 10; i++) {
//			listComparison[i]= new ArrayList<Comparison>();
//		}
//		//List<Comparison>[] listComparison= new ArrayList<Comparison>()[10];
//		Random r = new Random();
//		for (int i = 0; i < blocks.size(); i++) {		
//			//blocks_select.add(blocks.get(i));
//			//if(blocks.get(i).getNoOfComparisons()>2)
//			{
//				AbstractBlock b = blocks.get(i);
//				List<Comparison> c = b.getComparisons();
//				
//				for(Comparison com:c){
//					
//					com.teste=blocks.get(i).getBlockIndex();
//					com.sim=ebc.getSImilarity(com.getEntityId1(),com.getEntityId2());
//					int level=(int) Math.floor(com.sim*10);
//					
//					final List<Integer> commonBlockIndices =  entityIndex.getCommonBlockIndices(com.teste, com);
//					if (commonBlockIndices == null) {
//						continue;
//					}
////					if(commonBlockIndices.size()<1)
////						System.out.println(commonBlockIndices.size());
//					if(com.sim>= ((double)level*0.1) && com.sim<= ((double)(level+1)*0.1)){							
//						int temp=r.nextInt(nblocks[level]);
//						if((temp<500) || (temp<(nblocks[level]*0.1))){
//							//continue;
//							listComparison[level].add(com);
//						}
//						
//					}
//				}
//			}
//		}		
//		for (int j= 0; j < 10; j++) {
//			System.err.println("list size ---"+ listComparison[j].size());			
//		}		
//		return listComparison;
//	}
	
	   
	public static void main(String[] args) throws IOException, Exception {
		System.out.println( System.getProperty("user.home"));
		System.out.println( "agora sim blast " );
		String mainDirectory;
		String profilesPathA=null;
		String profilesPathB=null;
		String groundTruthPath = null;
		String[] args1 =new String[2];
		args1[0]="2";
		args1[1]="10K";
		 WeightingScheme ws = WeightingScheme.CHI_ENTRO;
	        //WeightingScheme ws = WeightingScheme.FISHER_ENTRO; // For dirty dataset use this test-statistic because of the low number of co-occurrence in the blocks (Fisher exact test vs. Chi-squared ~ approximated)
	        ThresholdWeightingScheme th_schme = ThresholdWeightingScheme.AM3;
		//args1[0]="acm";
		//args1[0]="dblp";
		
		switch(args1[0]){
		case "sint":	       
			mainDirectory = System.getProperty("user.home")+"/Dropbox/blocagem/bases/sintetica";
			profilesPathA =  mainDirectory+"/"+args1[1]+"profiles"	;	
			groundTruthPath =  mainDirectory+"/"+args1[1]+"IdDuplicates";	
			System.out.println("-----------"+mainDirectory);
			break;
		case "2":
			mainDirectory = System.getProperty("user.home")+"/Dropbox/blocagem/bases/base_clean_serializada";
			profilesPathA= mainDirectory+"/dblp";
			profilesPathB= mainDirectory+"/scholar";
			groundTruthPath =  mainDirectory+ "/groundtruth"; 
			break;
		case "3":
			mainDirectory = System.getProperty("user.home")+"/Dropbox/blocagem/bases/movies";

			profilesPathA= mainDirectory+"/token/dataset1_imdb";
			profilesPathB= mainDirectory+"/token/dataset2_dbpedia";
			groundTruthPath =  mainDirectory+ "/ground/groundtruth"; 
			break;
		case "4":
			mainDirectory = System.getProperty("user.home")+"/Dropbox/blocagem/bases/acm_cleanB";
			profilesPathA= mainDirectory+"//dblpB";
			profilesPathB= mainDirectory+"//acmB";
			//profilesPathB= mainDirectory+"/dataset2_gp";
			groundTruthPath =  mainDirectory+ "/groundtruth"; 
			break;
			
		case "5":
			//alto desbalanceamento descartar essa base
			mainDirectory = System.getProperty("user.home")+"/Dropbox/blocagem/bases/produtos/";
			profilesPathA =  mainDirectory+"/"+"amazon"	;	
			groundTruthPath =  mainDirectory+"/"+"groundtruth";	
			break;
		case "6":
			//alto desbalanceamento descartar essa base
			mainDirectory = System.getProperty("user.home")+"/Dropbox/blocagem/bases/prod/";
			profilesPathA =  mainDirectory+"/"+"dataset1_abt";	
			profilesPathB =  mainDirectory+"/"+"dataset2_buy";
			groundTruthPath =  mainDirectory+"/"+"groundtruth";	
			break;	
		case "7":
			mainDirectory = "/bkp/bases/blocagem_large";
			profilesPathA =  mainDirectory+"/"+"dataset1";	
			profilesPathB =  mainDirectory+"/"+"dataset2";
			groundTruthPath =  mainDirectory+"/"+"groundtruth";	
			break;
		}
	
		Set<IdDuplicates> duplicatePairs = (HashSet<IdDuplicates>) SerializationUtilities.loadSerializedObject(groundTruthPath);
		System.out.println("Existing duplicates\t:\t" + duplicatePairs.size());

		List<AbstractBlock> blocks = null;
		
		BilateralDuplicatePropagation adp =new BilateralDuplicatePropagation(groundTruthPath);
		
		List<EntityProfile>[] profiles ;
		if(profilesPathB != null){
			profiles = new List[2];
			profiles[0] = (List<EntityProfile>) SerializationUtilities.loadSerializedObject(profilesPathA);
			profiles[1] = (List<EntityProfile>) SerializationUtilities.loadSerializedObject(profilesPathB);
			
//			EntityProfile temp = profiles[0].get(0);
//			
//			Set<Attribute> att = temp.getAttributes();
//			
//			Iterator<Attribute> tt = att.iterator();
//			
//			//Attribute;
//			Attribute aaa = null;
//			while(tt.hasNext()){
//				aaa= tt.next();
//				System.out.print(aaa.getValue() +" " );
//			}
			//TokenBlocking imtb = new TokenBlocking(profiles);

			 //blocks = imtb.buildBlocks();					 
			AttributeClusteringBlockingEntropy blocking = new BlockBuilding.MemoryBased.AttributeClusteringBlockingEntropy(RepresentationModel.TOKEN_SHINGLING, profiles, 120, 3, true);
			blocks = blocking.buildBlocks();
			BlockStatistics bStats1 = new BlockStatistics(blocks, adp);
		    double[] values = bStats1.applyProcessing();
		    System.out.println("values 1 " + values[0] +" values 2 " + values[1] +" values 3" + values[2]);
			AbstractEfficiencyMethod blockPurging = new ComparisonsBasedBlockPurging(1.005);
			blockPurging.applyProcessing(blocks,adp,0);
//			
//			//////////			
			bStats1 = new BlockStatistics(blocks, adp);
		    values = bStats1.applyProcessing();
		    System.out.println("values 1 " + values[0] +" values 2 " + values[1] +" values 3" + values[2]);
			BlockFiltering bf = new BlockFiltering(0.9);
			bf.applyProcessing(blocks,adp);	
			bStats1 = new BlockStatistics(blocks, adp);
		    values = bStats1.applyProcessing();
		    System.out.println("values 1 " + values[0] +" values 2 " + values[1] +" values 3" + values[2]);
		    
		}else{
			profiles= new List[1];
			profiles[0] = (List<EntityProfile>) SerializationUtilities.loadSerializedObject(profilesPathA);
			TokenBlocking imtb = new TokenBlocking(profiles);
			
		    blocks = imtb.buildBlocks();			
			//ExtendedQGramsBlocking method = new ExtendedQGramsBlocking(0.95, 3, profiles);
			//QGramsBlocking imtb = new QGramsBlocking(6, profiles);
			//blocks = method.buildBlocks();			 
			//AbstractEfficiencyMethod blockPurging = new ComparisonsBasedBlockPurging(1.005);
			//blockPurging.applyProcessing(blocks);
			  
		        
		}
		String[] profilesPath;
		if(profilesPathB!=null){
			 profilesPath=new String[2];
			 profilesPath[0]=profilesPathA;
			 profilesPath[1]=profilesPathB;
		}else{
			 profilesPath=new String[1];
			 profilesPath[0]=profilesPathA;
		}
		int num_blocks=0;
		
		ExecuteBlockComparisons ebc = new ExecuteBlockComparisons(profilesPath);

		//for(profiles)
		//System.out.println(" numero comparações --> "+ num_blocks);
		adp =new BilateralDuplicatePropagation(groundTruthPath);
		WeightingScheme wScheme= WeightingScheme.CHI_ENTRO;
		
		MetaBlocking.EnhancedMetaBlocking.FastImplementations.RedefinedCardinalityNodePruning CNP= new MetaBlocking.EnhancedMetaBlocking.FastImplementations.RedefinedCardinalityNodePruning(wScheme,blocks.size()); 


		CNP.applyProcessing(blocks, adp,0);
		 
		BlockStatistics bStats1 = new BlockStatistics(blocks, adp);
		double[] values = bStats1.applyProcessing();
		System.out.println("final  1 " + values[0] +" values 2 " + values[1] +" values 3" + values[2]);
		
//		OnTheFlyMethods.FastImplementations.RedefinedWeightedNodePruning b_wnp = new OnTheFlyMethods.FastImplementations.RedefinedWeightedNodePruning(adp, ws, th_schme, blocks.size());
//		b_wnp.applyProcessing(blocks, adp, ebc);
//	 BlastWeightedNodePruning b_wnp = new BlastWeightedNodePruning(adp, ws, th_schme, blocks.size());
//	 b_wnp.applyProcessing(blocks);
//	     double[] values = CNP1.getPerformance();
//
//	    System.out.println("pc: " + values[0]);
//	    System.out.println("pq: " + values[1]);
//	    System.out.println("f1: " + (2 * values[0] * values[1]) / (values[0] + values[1]));
//	    System.out.println("blocks " + blocks.size() +" blocks " );
//	    int count=0;
	   
//	        if(blocks.isEmpty()){
//	        	
//	        	System.out.println("errooooo");
//	        	return;
//	        }
//	        BlockStatistics bStats1 = new BlockStatistics(blocks, adp);
//	        values = bStats1.applyProcessing();
//	        System.out.println("values blast " + values[0] +" values 2 " + values[1] +" values 3" + values[2]);
		//            System.out.println("\n\n\n\n\n======================= Supervised CEP =======================");

		Classifier[] classifiers = getSupervisedWepClassifiers();
		//            SupervisedCEP scep = new SupervisedCEP(classifiers.length, blocks, duplicatePairs);
		//            for (int j = 0; j < ITERATIONS; j++) {
		//                scep.applyProcessing(j, classifiers, ebc);
		//            }
		//            scep.printStatistics();
		//
		//		            System.out.println("\n\n\n\n\n======================= Supervised CNP =======================");
		//		            classifiers = getSupervisedCnpClassifiers();
		//		            SupervisedCNP scnp = new SupervisedCNP(classifiers.length, blocks, duplicatePairs);
		//		            for (int j = 0; j < 5; j++) {
		//		                scnp.applyProcessing(j, classifiers, ebc);
		////		                BlockStatistics blockStats = new BlockStatistics(blocks, new BilateralDuplicatePropagation(mainDirectory+ "/groundtruth"));
		////		     		   double teste[]=blockStats.applyProcessing();
		////		     		   System.out.println("------------" +teste[0] + "  "+ teste[1] + "  "+ teste[0]);
		//		            }
		//		            scnp.printStatistics();
//		Random r=new Random();
//		int n=r.nextInt(100);
//
//		BufferedWriter writer1 = new BufferedWriter(new FileWriter(System.getProperty("user.home")+"/Dropbox/blocagem/saida50K_classificador1"+profilesPathA.split("/")[profilesPathA.split("/").length-1]));
//		BufferedWriter writer2 = new BufferedWriter(new FileWriter(System.getProperty("user.home")+"/Dropbox/blocagem/saida50K_classificador2"+profilesPathA.split("/")[profilesPathA.split("/").length-1]));
//		BufferedWriter writer3 = new BufferedWriter(new FileWriter(System.getProperty("user.home")+"/Dropbox/blocagem/saida50K_classificador3"+profilesPathA.split("/")[profilesPathA.split("/").length-1]));
//		BufferedWriter writer4 = new BufferedWriter(new FileWriter(System.getProperty("user.home")+"/Dropbox/blocagem/saida50K_classificador4"+profilesPathA.split("/")[profilesPathA.split("/").length-1]));
//
//       
//		
//		classifiers = getSupervisedWepClassifiers();
//		SupervisedWEP swep;
//
//		//new EntityIndex(blocks).enumerateBlocks(blocks);;	
//		File f=new File("/tmp/lock");
//		f.delete();
//
//		System.out.println("\n\n\n\n\n======================= Supervised WEP =======================" + " " + duplicatePairs.size());
//		int i=1,j=5;
//		//for (int i = 1; i <= 2;i++)
//		{
//			swep = new SupervisedWEP(classifiers.length, blocks, duplicatePairs,ebc);
//
//			//blockHash.produceHash(blocks, ebc);
//			int tamanho = 100;
//			//while(tamanho<=1000)
//			{
//
//				writer1.write("level "+tamanho +"\n");
//				writer2.write("level "+tamanho +"\n");
//				writer3.write("level "+tamanho+"\n");
//				writer4.write("level "+tamanho+"\n");
//
//				for (j = 0;j< 1; j++) 
//				{
//					swep.applyProcessing(0,j, classifiers, ebc, tamanho, writer1,writer2,writer3,writer4,i,profilesPathA.split("/")[profilesPathA.split("/").length-1]);
//
//					writer1.flush();
//					writer2.flush();
//					writer3.flush();
//					writer4.flush();
//				}
//				swep.printStatistics();
//				//swep.printStatisticsB(writer);
//				System.out.println("size of level : "+ tamanho);
//
//
//				if(tamanho==5)
//					tamanho=10;
//				else if(tamanho==10)
//					tamanho=50;
//				else if(tamanho==50)
//					tamanho*=2;
//				else if(tamanho==100)
//					tamanho=500;
////				
//				else if( tamanho==500)
//					tamanho=1000;
//				else
//				if(tamanho==1000)
//					tamanho+=tamanho;
////				else tamanho*=tamanho;
//			//	ebc.temp_limiar+=0.1;
//
//
//			}
//		}
//		writer1.close();
//		writer2.close();
//		writer3.close();
	}
	//  }
}
