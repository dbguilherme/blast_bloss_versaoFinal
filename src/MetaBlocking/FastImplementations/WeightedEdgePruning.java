package MetaBlocking.FastImplementations;

import DataStructures.AbstractBlock;
import MetaBlocking.WeightingScheme;

import java.util.List;
import java.util.Set;

import BlockProcessing.ComparisonRefinement.AbstractDuplicatePropagation;

/**
 * @author stravanni
 */
public class WeightedEdgePruning extends AbstractFastMetablocking {

    protected double noOfEdges;
    double max_min;
    int ii;

    public WeightedEdgePruning(WeightingScheme scheme) {
        this("Fast Weighted Edge Pruning (" + scheme + ")", scheme);
        max_min = Double.MAX_VALUE;
        ii = 0;
    }

    protected WeightedEdgePruning(String description, WeightingScheme scheme) {
        super(description, scheme);
        nodeCentric = false;
    }

    protected void processArcsEntity(int entityId) {
        validEntities.clear();
        final int[] associatedBlocks = entityIndex.getEntityBlocks(entityId, 0);
        if (associatedBlocks.length == 0) {
            return;
        }

        for (int blockIndex : associatedBlocks) {
            double blockComparisons = cleanCleanER ? bBlocks[blockIndex].getNoOfComparisons() : uBlocks[blockIndex].getNoOfComparisons();
            setNormalizedNeighborEntities(blockIndex, entityId);
            for (int neighborId : neighbors) {
                if (flags[neighborId] != entityId) {
                    counters[neighborId] = 0;
                    flags[neighborId] = entityId;
                }

                counters[neighborId] += 1 / blockComparisons;
                validEntities.add(neighborId);
            }
        }
    }

    protected void processArcs_entro_Entity(int entityId) {
        validEntities.clear();
        final int[] associatedBlocks = entityIndex.getEntityBlocks(entityId, 0);
        if (associatedBlocks.length == 0) {
            return;
        }

        for (int blockIndex : associatedBlocks) {
            double blockComparisons = cleanCleanER ? bBlocks[blockIndex].getNoOfComparisons() : uBlocks[blockIndex].getNoOfComparisons();
            setNormalizedNeighborEntities(blockIndex, entityId);
            for (int neighborId : neighbors) {
                if (flags[neighborId] != entityId) {
                    counters[neighborId] = 0;
                    flags[neighborId] = entityId;
                }

                counters[neighborId] += 1 / blockComparisons;
                counters_entro[neighborId] += entityIndex.getEntropyBlock(blockIndex);

                validEntities.add(neighborId);
            }
        }
    }

//    protected void processCHI_entro_Entity(int entityId) {
//        validEntities.clear();
//        final int[] associatedBlocks = entityIndex.getEntityBlocks(entityId, 0);
//        if (associatedBlocks.length == 0) {
//            return;
//        }
//
//        for (int blockIndex : associatedBlocks) {
//            //double blockComparisons = cleanCleanER ? bBlocks[blockIndex].getNoOfComparisons() : uBlocks[blockIndex].getNoOfComparisons();
//            setNormalizedNeighborEntities(blockIndex, entityId);
//            for (int neighborId : neighbors) {
//                if (flags[neighborId] != entityId) {
//                    counters[neighborId] = 0;
//                    counters_entro[neighborId] = 0;
//                    flags[neighborId] = entityId;
//                }
//
//                //counters[neighborId] += 1 / blockComparisons;
//                counters[neighborId]++;
//                counters_entro[neighborId] += entityIndex.getEntropyBlock(blockIndex);
//
//                validEntities.add(neighborId);
//            }
//        }
//    }

    protected void processEntity(int entityId) {
    	
    	//####
    	for (int validEntries : validEntities) {
    	    flags[validEntries] = -1;
    	    counters[validEntries] = 0;
        	counters_entro[validEntries] = 0;
    	}
    	    	   	
    	validEntities.clear();
    	validEntitiesNeighbor.clear();

        final int[] associatedBlocks = entityIndex.getEntityBlocks(entityId, 0);
        if (associatedBlocks.length == 0) {
            return;
        }

        for (int blockIndex : associatedBlocks) {
            setNormalizedNeighborEntities(blockIndex, entityId);
            for (int neighborId : neighbors) {
                if (flags[neighborId] != entityId) {
                    counters[neighborId] = 0;
                    counters_entro[neighborId] = 0;
                    flags[neighborId] = entityId;
                }

                counters[neighborId]++;
                counters_entro[neighborId] += entityIndex.getEntropyBlock(blockIndex);
                //System.out.println("counters_entro: " + counters_entro[neighborId]);
                validEntities.add(neighborId);
                //###
                validEntitiesNeighbor.add(blockIndex);
            }
        }
    }

    protected void pruneEdges(List<AbstractBlock> newBlocks, AbstractDuplicatePropagation adp) {
        int limit = cleanCleanER ? datasetLimit : noOfEntities;
        if (weightingScheme.equals(WeightingScheme.ARCS)) {
            for (int i = 0; i < limit; i++) {
                processArcsEntity(i);
                verifyValidEntities(i, newBlocks);
            }
        } else {
            for (int i = 0; i < limit; i++) {
                processEntity(i);
                verifyValidEntities(i, newBlocks);
            }
        }
    }

    @Override
    protected void setThreshold() {
        noOfEdges = 0;
        threshold = 0;

        int limit = cleanCleanER ? datasetLimit : noOfEntities;
        if (weightingScheme.equals(WeightingScheme.ARCS)) {
            for (int i = 0; i < limit; i++) {
                processArcsEntity(i);
                updateThreshold(i);
            }
        } else {
            for (int i = 0; i < limit; i++) {
                processEntity(i);
                updateThreshold(i);
            }
        }

        threshold /= noOfEdges;
        //threshold = threshold/2;
        //threshold /= (ii*2);
        //System.out.println("\n\n\nglobal threshold: " + threshold);
        //threshold = max_min;
        //System.out.println("\n\n\nglobal max_min  : " + max_min);
    }

    protected void updateThreshold(int entityId) {
//        noOfEdges += validEntities.size();
//        double max = 0;
//        for (int neighborId : validEntities) {
//            //threshold += getWeight(entityId, neighborId);
//            max = Math.max(getWeight(entityId, neighborId),max);
//        }
//        //System.out.println("max local: " + max);
//        if (max > 0.0) {
//            //System.out.println("max local: " + max);
//            //ii ++;
//            //threshold += max;
//            max_min = Math.min(max_min, max);
//        }
//
////        if (max > 0.0 && max < 1.0) {
////            System.out.println("max local: " + max);
////        }
//        //max_min += max;
        noOfEdges += validEntities.size();
        for (int neighborId : validEntities) {
            threshold += getWeight(entityId, neighborId);
        }
    }

    protected void verifyValidEntities(int entityId, List<AbstractBlock> newBlocks) {
        retainedNeighbors.clear();
        if (!cleanCleanER) {
            for (int neighborId : validEntities) {
                double weight = getWeight(entityId, neighborId);
                if (threshold <= weight) {
                    retainedNeighbors.add(neighborId);
                }
            }
            addDecomposedBlock(entityId, retainedNeighbors, newBlocks);
        } else {
            if (entityId < datasetLimit) {
                for (int neighborId : validEntities) {
                    double weight = getWeight(entityId, neighborId);
                    if (threshold <= weight) {
                        retainedNeighbors.add(neighborId - datasetLimit);
                    }
                }
                addDecomposedBlock(entityId, retainedNeighbors, newBlocks);
            } else {
                for (int neighborId : validEntities) {
                    double weight = getWeight(entityId, neighborId);
                    if (threshold <= weight) {
                        retainedNeighbors.add(neighborId);
                    }
                }
                addReversedDecomposedBlock(entityId - datasetLimit, retainedNeighbors, newBlocks);
            }
        }
    }


	

	
}
