/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    Copyright (C) 2015 George Antony Papadakis (gpapadis@yahoo.gr)
 */

package MetaBlocking.EnhancedMetaBlocking.FastImplementations;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

import BlockProcessing.ComparisonRefinement.AbstractDuplicatePropagation;
import Comparators.ComparisonWeightComparator;
import DataStructures.AbstractBlock;
import DataStructures.BilateralBlock;
import DataStructures.Comparison;
import MetaBlocking.WeightingScheme;
import Utilities.ComparisonIterator;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;


/**
 * @author gap2
 */
public class RedefinedCardinalityNodePruning extends MetaBlocking.FastImplementations.CardinalityNodePruning {

    protected Set<Comparison>[] nearestEntities;

    public RedefinedCardinalityNodePruning(WeightingScheme scheme) {
        this("Fast Redundancy Cardinality Node Pruning (" + scheme + ")", scheme);
    }

    public RedefinedCardinalityNodePruning(WeightingScheme scheme, double totalBlcoks) {
        this("Fast Redundancy Cardinality Node Pruning (" + scheme + ")", scheme);
        this.totalBlocks = totalBlcoks;
    }

    protected RedefinedCardinalityNodePruning(String description, WeightingScheme scheme) {
        super(description, scheme);
    }

    protected boolean isValidComparison(int entityId, Comparison comparison) {
        int neighborId = comparison.getEntityId1() == entityId ? comparison.getEntityId2() : comparison.getEntityId1();
        if (cleanCleanER && entityId < datasetLimit) {
            neighborId += datasetLimit;
        }

        if (nearestEntities[neighborId] == null) {
            return true;
        }

        if (nearestEntities[neighborId].contains(comparison)) {
            return entityId < neighborId;
        }

        return true;
    }

    
    
 //###
    
    protected int noOfAttributes;
    protected int noOfClassifiers;
    protected double noOfBlocks;
    protected double validComparisons;
    protected double[] comparisonsPerBlock;
    protected double[] nonRedundantCPE;
    protected double[] redundantCPE;
    ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    List<String> classLabels;
    AbstractDuplicatePropagation adp;
    
    private void getStatistics(List<AbstractBlock> blocks) {
        noOfBlocks = blocks.size();
        System.out.println("BLOCK SIZE "+ (blocks.size() + 1));
        validComparisons = 0;
        int noOfEntities = entityIndex.getNoOfEntities();
        
        redundantCPE = new double[noOfEntities];
        nonRedundantCPE = new double[noOfEntities];
        comparisonsPerBlock = new double[(int)(blocks.size() + 1)];
        for (AbstractBlock block : blocks) {
        	block.setNoOfComparisons(((BilateralBlock)block).getIndex1Entities().length* ((BilateralBlock)block).getIndex2Entities().length);
        //	System.out.println("block.getNoOfComparisons()  " +block.getNoOfComparisons() +" block.getBlockIndex()  " + block.getBlockIndex());
            comparisonsPerBlock[block.getBlockIndex()] = block.getNoOfComparisons();
            
            ComparisonIterator iterator = block.getComparisonIterator();
            while (iterator.hasNext()) {
                Comparison comparison = iterator.next();
                    
                int entityId2 = comparison.getEntityId2()+entityIndex.getDatasetLimit();
                redundantCPE[comparison.getEntityId1()]++;
                redundantCPE[entityId2]++;
                    
                if (!entityIndex.isRepeated(block.getBlockIndex(), comparison)) {
                    validComparisons++;
                    nonRedundantCPE[comparison.getEntityId1()]++;
                    nonRedundantCPE[entityId2]++;
                }
            }
        }
    }
    
    private void getAttributes() {
        
        attributes.add(new Attribute("ECBS"));
        attributes.add(new Attribute("RACCB"));
        attributes.add(new Attribute("JaccardSim"));
        attributes.add(new Attribute("NodeDegree1"));
        attributes.add(new Attribute("NodeDegree2"));
        attributes.add(new Attribute("teste weight"));
       attributes.add(new Attribute("teste weight"));
      // attributes.add(new Attribute("teste weight"));
       //attributes.add(new Attribute("teste weight"));
        classLabels = new ArrayList<String>();
        classLabels.add("0");
        classLabels.add("1");
        
        Attribute classAttribute = new Attribute("class", classLabels);
        attributes.add(classAttribute);
        noOfAttributes = attributes.size();
    }
    
    @Override
    protected void pruneEdges(List<AbstractBlock> newBlocks, AbstractDuplicatePropagation adp) {
        nearestEntities = new Set[noOfEntities];
        topKEdges = new PriorityQueue<Comparison>((int) (2 * threshold), new ComparisonWeightComparator());
        this.adp=adp;
        getStatistics(newBlocks);
        getAttributes();
        
    	int matchingInstances=3000;
  	    HashSet<Comparison> trainingSet = new HashSet<Comparison>(4*matchingInstances);
        Instances trainingInstances = new Instances("trainingSet", attributes, 2*matchingInstances);
        trainingInstances.setClassIndex(noOfAttributes - 1);
        newBlocks.clear();
        
        if (weightingScheme.equals(WeightingScheme.ARCS)) {
            for (int i = 0; i < noOfEntities; i++) {
                processArcsEntity(i);
                verifyValidEntities(i,trainingInstances);
            }
        } else {
            for (int i = 0; i < noOfEntities; i++) {
                processEntity(i);
                verifyValidEntities(i,trainingInstances);
            }
        }
        try{
        	ArffSaver saver = new ArffSaver();
            saver.setInstances(trainingInstances);
            saver.setFile(new File("/tmp/test.arff"));
            saver.setDestination(new File("/tmp/test.arff"));   // **not** necessary in 3.5.4 and later
            saver.writeBatch();
        }catch(Exception e){
        	System.err.println("ERRO SALVAMENTO ARQUIVO" );
        	e.getStackTrace();
        }
        			
        retainValidComparisons(newBlocks);
    }

    protected void retainValidComparisons(List<AbstractBlock> newBlocks) {
        final List<Comparison> retainedComparisons = new ArrayList<>();
        for (int i = 0; i < noOfEntities; i++) {
            if (nearestEntities[i] != null) {
                retainedComparisons.clear();
                for (Comparison comparison : nearestEntities[i]) {
                    if (isValidComparison(i, comparison)) {
                        retainedComparisons.add(comparison);
                    }
                }
                addDecomposedBlock(retainedComparisons, newBlocks);
            }
        }
    }

   // @Override
    protected void verifyValidEntities(int entityId,  Instances trainingInstances) {
        if (validEntities.isEmpty()) {
            return;
        }

        topKEdges.clear();
        minimumWeight = Double.MIN_VALUE;
        Iterator<Integer> it = validEntitiesNeighbor.iterator();
        for (int neighborId : validEntities) {
        	// System.out.println("comparison A" + entityId +"   "+ neighborId);
        	// if(entityId==2516)
        	//	 System.out.println("2516 ---");
            double weight = getWeight(entityId, neighborId);
            int blockId=it.next();
            
            if(neighborId==6792)
        		System.out.println("ok");
            if (weight < minimumWeight) {
                continue;
            }
          
            Comparison comparison = getComparison(entityId, neighborId);
           
            comparison.setUtilityMeasure(weight);
            comparison.blockId=blockId;
            topKEdges.add(comparison);
            if (threshold < topKEdges.size()) {
                Comparison lastComparison = topKEdges.poll();
                minimumWeight = lastComparison.getUtilityMeasure();
            }
        }
        
        nearestEntities[entityId] = new HashSet<Comparison>(topKEdges);
        Iterator<Comparison> itb = nearestEntities[entityId].iterator();
        while(itb.hasNext()){
        	Comparison c=itb.next();
        	int neighborId_clean;
        	int neighborId = c.getEntityId1() == entityId ? c.getEntityId2() : c.getEntityId1();
        	neighborId_clean=neighborId;
        	if(neighborId_clean==6792 || neighborId==6792)
        		System.out.println("ok");
            if (cleanCleanER && entityId < datasetLimit) {
                neighborId += datasetLimit;
            }
//
//            if (nearestEntities[neighborId] == null) {
//                continue;
//            }
//
//            if (nearestEntities[neighborId].contains(c)) {
//                if(! (entityId < neighborId))
//                	continue;
//            }
           
           // System.out.println(entityId +" "+ neighborId);
//            if(entityId>datasetLimit){
//        		int temp=neighborId_clean;
//        		neighborId=entityId;
//        		entityId=temp;
//        	}
            Comparison comp = new Comparison(true, entityId, neighborId_clean);
        	
        	
        	final List<Integer> commonBlockIndices = entityIndex.getCommonBlockIndices(c.blockId, comp);
			if(commonBlockIndices==null)
				continue;
			
			double[] instanceValues = new double[8];


			double ibf1 = Math.log(noOfBlocks/entityIndex.getNoOfEntityBlocks(entityId, 0));
			double ibf2 = Math.log(noOfBlocks/entityIndex.getNoOfEntityBlocks(neighborId, 0));

			instanceValues[0] = commonBlockIndices.size()*ibf1*ibf2;

			double raccb = 0;
			for (Integer index1 : commonBlockIndices) {
				raccb += 1.0 / comparisonsPerBlock[index1];
			}
			if (raccb < 1.0E-6) {
				raccb = 1.0E-6;
			}
			instanceValues[1] = raccb;
			instanceValues[2] =commonBlockIndices.size() / (redundantCPE[entityId] + redundantCPE[neighborId] - commonBlockIndices.size());
			instanceValues[3] = nonRedundantCPE[entityId];
			instanceValues[4] = nonRedundantCPE[neighborId];
			//   	instanceValues[5] =	ebc.getSimilarityAttribute(c.getEntityId1(), c.getEntityId2());
			
			instanceValues[5]= neighborId;
			instanceValues[6]=entityId;//c.getUtilityMeasure(); 
					//(Math.sqrt(Math.pow(averageWeight[entityId], 2) + Math.pow(averageWeight[neighborId], 2)) / 4) *  getWeight(c.getEntityId1(), c.getEntityId2()+datasetLimit);

			instanceValues[7] = adp.isSuperfluous(c)==true?0:1;//adp.isSuperfluous(getComparison(c.getEntityId1(), c.getEntityId2()+datasetLimit))?1:0;
			
			Instance newInstance = new DenseInstance(1.0, instanceValues);
			newInstance.setDataset(trainingInstances);
			trainingInstances.add(newInstance);     
//			for (int i = 5; i < instanceValues.length-1; i++) {
//				System.out.print(instanceValues[i] +" ");
//			}
//        	System.out.println();
//			if(instanceValues[6]!=instanceValues[5])
//				System.out.println("erro");
//			else
//				System.out.print("...");
        }
        
    }
}
