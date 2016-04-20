///////////////////////////////////////////////////////////////////////////////
//
//Title:            DecisionTreeImpl.java
//Main Class:       HW4.java
//Files:            DataSet.java, DecisionTree.java, DecTreeNode.java
//                  , Instance.java
//Semester:         Spring 2016
//
//Author:           Han Jiang - hjiang73@wisc.edu
//CS Login:         hjiang  
//Lecturer's Name:  Collin Engstrom  
//
///////////////////////////////////////////////////////////////////////////////
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 * 
 * You must add code for the 5 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
	private DecTreeNode root;
	private List<String> labels; // ordered list of class labels
	private List<String> attributes; // ordered list of attributes
	private Map<String, List<String>> attributeValues; // map to ordered
	// discrete values taken
	// by attributes
	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary
		// this is void purposefully
	}

	/**
	 * Build a decision tree given only a training set.
	 * 
	 * @param train: the training set
	 */
	DecisionTreeImpl(DataSet train) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		//return a tree
		root = buildtree(train, train.instances, null, train.attributes, -1);

	}

	/**
	 * The helper method which uses ID3 algorithm to implement decision tree
	 * 
	 * @param train: the training set, the list of instances, the list of parent instances, 
	 * the list of attributes, the index of parents's attribute
	 * @return a DecTreeNode pointing to the root of decision tree
	 */
	private DecTreeNode buildtree(DataSet train, List<Instance>instances, List<Instance> parents, List<String> attributes, int parentsattr){
		//if empty(instances) then return(default)
		if (instances.isEmpty()) {
			Integer label  = labels.indexOf(majorityValue(parents, train));
			return new DecTreeNode(label, null, parentsattr, true);

		}

		//if instances have same lable y, then return y
		if (allExamplesHaveSameLabel(instances)) {
			Integer label = instances.get(0).label;
			return new DecTreeNode(label, null, parentsattr, true);
		}
		//if empty(questions) Then return (majority votes in examples)
		if(attributes.isEmpty()){
			Integer label =  labels.indexOf(majorityValue(instances, train));
			return new DecTreeNode(label, null, parentsattr, true);
		}

		//else find the attribute which can maximize the information gain
		//Recursive function
		//Find the attribute which has the largest information gain
		String MaxAtt = chooseAttribute(train, instances, attributes);
		Integer MaxInt = train.attributes.indexOf(MaxAtt);
		//Find the majority votes of all instances
		int[] labels = new int[train.labels.size()];
		Iterator<Instance> intitr = instances.iterator();
		while(intitr.hasNext()){
			labels[intitr.next().label]++;  	
		}
		//The index of the label
		int MaxLab = 0;
		for(int i =0; i<train.labels.size();i++){
			if(labels[i] > labels[MaxLab]){
				MaxLab = i;
			}
		}
		//Build a new node based on the attribute with largest infogain
		DecTreeNode bestnode = new DecTreeNode(MaxLab,MaxInt,parentsattr,false);
		//Delete the attribute and return remain attributes
		List<String> leftattributes = new ArrayList<String>();
		Iterator<String> leftitr = attributes.iterator();
		while(leftitr.hasNext()){
			String left = leftitr.next();
			if(!left.equals(MaxAtt)){
				leftattributes.add(left);
			}

		}
		//The ith child is built by calling buildtree
		for(int n =0;n<train.attributeValues.get(MaxAtt).size();n++){
			List<Instance> leftinstances= new ArrayList<Instance>();
			Iterator<Instance> _leftitr = instances.iterator();
			//remain instances
			while(_leftitr.hasNext()){
				Instance _left = _leftitr.next();
				if(_left.attributes.get(MaxInt) == n){
					leftinstances.add(_left);
				}
			}
			//create child nodes
			DecTreeNode child = buildtree(train,leftinstances,instances,leftattributes,n);
			bestnode.addChild(child);
		}
		return bestnode;
	}

	/**
	 * Find over-all majority vote of parent nodes 
	 * @param Dataset train
	 * @param List of all attributes
	 * @return String which has largest information gain
	 */
	private String majorityValue(List<Instance> parents, DataSet data){

		int[] labelscounter = new int[data.labels.size()];
		Iterator<Instance> insitr = parents.iterator();
		while(insitr.hasNext()){
			Instance tmp = insitr.next();
			labelscounter[tmp.label]++;
		}
		int majorityvote = labelscounter[0];
		int index = 0;
		for(int i =0; i<labelscounter.length;i++){
			if(labelscounter[i]>majorityvote){
				index = i;
			}
		}
		return data.labels.get(index);
	}

	/**
	 * Return the attribute that can result in pure child nodes 
	 * @param Dataset train
	 * @param List of all attributes
	 * @return String which has largest information gain
	 */

	private String chooseAttribute(DataSet dataset, List<Instance> instances, List<String> attributes) {
		double greatestGain = 0.0;
		String attributeWithGreatestGain = attributes.get(0);
		Iterator<String> itr = attributes.iterator();
		while(itr.hasNext()){
			String attr = itr.next();
			double gain = calculateGainFor(dataset, instances, attr);
			if (gain > greatestGain) {
				greatestGain = gain;
				attributeWithGreatestGain = attr;
			}
		}
		return attributeWithGreatestGain;
	}


	/**
	 * Use entropy to calculate the information gain for each attributes 
	 * @param Dataset train
	 * @param List of all attributes
	 * @return String which has largest information gain
	 */
	private double calculateGainFor(DataSet ds, List<Instance> instances, String attribute){
		int totalinstances = instances.size();
		int totallabel = ds.labels.size();
		//Calculate H(Y) = ∑(i=1,K)-Pr(Y=yi)log2Pr(Y=yi)
		//Calculate Pr(Y=yi)
		double pro = 0.0;
		List<Double> pros = new ArrayList<Double>();
		for(int i =0; i<totallabel;i++){
			int tmplabel = labels.indexOf((ds.labels.get(i)));
			int count = 0;
			for(int j =0;j<totalinstances;j++){
				if(tmplabel==instances.get(j).label){
					count++;
				}
			}
			pro = (double)count/totalinstances;
			pros.add(pro);
		}
		Iterator<Double> itr  = pros.iterator();
		double h_y = 0.0;
		while (itr.hasNext()){
			double tmppro = itr.next();
			if(tmppro==0){
			}
			else{
				h_y = h_y - tmppro* Math.log(tmppro)/Math.log(2);
			}
		}

		//Calculate Conditional entropy
		//H(Y|X) = ∑(i=1,K)-Pr(Y=yi|X=attribute)log2Pr(Y=yi|X=attribute)
		//Pr(X=x)
		int[][] attribute_label = new int [ds.attributeValues.get(attribute).size()][ds.labels.size()];

		int attrindex = ds.attributes.indexOf(attribute);	

		for(int i =0;i<instances.size();i++){
			Instance tmp = instances.get(i);
			int labelindex = (tmp.label);
			attribute_label[instances.get(i).attributes.get(attrindex)][labelindex]++;
			//ds.attributeValues.get(attribute).indexOf(String.valueOf(instances.get(i).attributes.get(attrindex)+1))
		}

		List<Integer> sumvalue = new ArrayList<Integer>();
		List<Double> conpros = new ArrayList<Double>();
		for(int j =0;j<attribute_label.length;j++){
			int sumofvalue = 0;
			for(int k =0;k<attribute_label[j].length;k++){
				sumofvalue = sumofvalue+attribute_label[j][k];
			}
			sumvalue.add(sumofvalue);
			double conpro = (double)sumofvalue/instances.size();
			conpros.add(conpro);
		}

		List<Double> h_y_xs = new ArrayList<Double>();
		for(int j =0;j<attribute_label.length;j++){
			double h_y_x =0.0;
			for(int k =0;k<attribute_label[j].length;k++){

				if(attribute_label[j][k]==0){


				}
				else { double _conpro = (double)attribute_label[j][k]/sumvalue.get(j);

				h_y_x = h_y_x-(double)_conpro*Math.log(_conpro)/Math.log(2);

				}
			}
			h_y_xs.add(h_y_x);

		}

		double conentropy = 0.0;
		for(int m =0;m<h_y_xs.size();m++){
			conentropy += conpros.get(m)*h_y_xs.get(m);

		}
		//return infogain
		double infogain = h_y-conentropy;
		return infogain;
	}

	/**
	 * Determines if all instances have same label
	 * @param instances List of Instance objects
	 * @return true if all instances have the same label, false otherwise
	 */
	private boolean allExamplesHaveSameLabel(List<Instance> inst){
		Integer label =inst.get(0).label;
		Iterator<Instance> iter = inst.iterator();
		while(iter.hasNext()){
			Instance instance = iter.next();
			if(!(instance.label.equals(label))){
				return false;
			}
		}
		return true;
	}


	/**
	 * Build a decision tree given a training set then prune it using a tuning
	 * set.
	 * 
	 * @param train: the training set
	 * @param tune: the tuning set
	 */

	DecisionTreeImpl(DataSet train, DataSet tune) {
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		//build a decision tree
		root = buildtree(train, train.instances, null, train.attributes, -1);
		//prune it
		prune(train,tune,root);

	}

	/**
	 * The auxiliary method which prunes the original tree based on tune set
	 * set.
	 * @param train: the training set
	 * @param tune: the tuning set
	 * @param DecTreeNode node
	 * @return a DecTreeNode pointing to the post-pruned tree
	 */
	private void prune(DataSet train, DataSet tune,DecTreeNode node){
		double accuracy_T = accuracy(tune);

		List<Double> accuracys = new ArrayList<Double>();
		List<DecTreeNode> rootnodes = new ArrayList<DecTreeNode>();

		if (node.terminal){
			return;
		}
		if(node.children.isEmpty()||node.children == null){
			return;
		}
		//traverse every node in the tree and calculate the accuracy, then store it in the list
		else{	
			node.terminal = true;
			rootnodes.add(node);
			double accuracy =accuracy(tune);
			accuracys.add(accuracy);
			node.terminal = false;
		}
		//recurse
		for (DecTreeNode currNode : node.children) {
			prune(train,tune,currNode);
		}
		//find the node with the largest accuracy
		//The index will always return the first(smaller) subtree
		//if they have same accuracy
		int index = accuracys.indexOf((Collections.max(accuracys)));
		node = rootnodes.get(index);
		//compare the subtree to the original one
		if(Collections.max(accuracys)>=accuracy_T){
			node.children = null;
			node.terminal =true;
		}
	}


	/**
	 * The auxiliary method to calculate the accuracy of a tree
	 * @param test: the testing set
	 * @return double accuracy
	 */

	private double accuracy(DataSet test){
		List<Integer> correcttest = getinslabels(test);
		List<Integer> predicttest = predictlabels(test);
		if (correcttest.size() != predicttest.size()) {
			return 0;
		} else {
			int right = 0, wrong = 0;
			for (int i = 0; i < predicttest.size(); i++) {
				if (predicttest.get(i) == null) {
					wrong++;
				} else if (predicttest.get(i).equals(correcttest.get(i))) {
					right++;
				} else {
					wrong++;
				}
			}
			return right * 1.0 / (right + wrong);	
		}

	}

	/**
	 * The auxiliary method to get the labels of all instances of a dataset
	 * @param DataSet ds
	 * @return List of all labels
	 */
	private List<Integer> getinslabels(DataSet ds){
		List<Integer> labelofds = new ArrayList<Integer>();
		List<Instance> instances = ds.instances;
		Iterator<Instance> insitr = instances.iterator();
		while(insitr.hasNext()){
			labelofds.add(insitr.next().label);
		}
		return labelofds;
	}

	/**
	 * The auxiliary method to get the all predicted labels of a dataset
	 * @param DataSet ds
	 * @return List of all predicted labels
	 */
	private List<Integer> predictlabels(DataSet ds){
		List<Integer> predictlabel = new ArrayList<Integer>();

		for (Instance instance : ds.instances) {
			String _label = classify(instance);
			int label = ds.labels.indexOf(_label);
			predictlabel.add(label);
		}
		return predictlabel;
	}

	@Override

	/**
	 * The classify method will return the predicted label when given an instance
	 * based on the tree which we built
	 * @param Instance instance
	 * @return String - predicted label
	 */

	public String classify(Instance instance) {
		return predict(this.root, instance, this.labels,this.attributes,this.attributeValues);
	}

	/**
	 * The auxiliary method to help determine the label of a input instance, 
	 * @param Instance instance
	 * @param DecTreeNode - root node of a tree
	 * @param List<labels> - list of all labels in a dataset
	 * @return String - predicted label
	 */ 
	private String predict(DecTreeNode node, Instance instance, List<String> labels,
			List<String> attributes,Map<String, List<String>> attributeValues){
		while(true){
			if(node.terminal){
				break;
			}
			else{
				DecTreeNode found = null;
				for(DecTreeNode child : node.children){
					if(child.parentAttributeValue.equals(instance.attributes.get(node.attribute))){
						found = child;
						break;
					}
				}
				node = found;
			}
		}
		return labels.get(node.label);
	}

	@Override
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {
		printTreeNode(root, null, 0);
	}

	/**
	 * Prints the subtree of the node
	 * with each line prefixed by 4 * k spaces.
	 */
	public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < k; i++) {
			sb.append("    ");
		}
		String value;
		if (parent == null) {
			value = "ROOT";
		} else{
			String parentAttribute = attributes.get(parent.attribute);
			value = attributeValues.get(parentAttribute).get(p.parentAttributeValue);
		}
		sb.append(value);
		if (p.terminal) {
			sb.append(" (" + labels.get(p.label) + ")");
			System.out.println(sb.toString());
		} else {
			sb.append(" {" + attributes.get(p.attribute) + "?}");
			System.out.println(sb.toString());
			for(DecTreeNode child: p.children) {
				printTreeNode(child, p, k+1);
			}
		}
	}

	@Override
	public void rootInfoGain(DataSet train) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;

		for(int i =0;i<this.attributes.size();i++){
			System.out.print(this.attributes.get(i)+" ");
			System.out.format("%.5f",calculateGainFor(train, train.instances ,this.attributes.get(i)));
			System.out.println();
		}
	}
}
