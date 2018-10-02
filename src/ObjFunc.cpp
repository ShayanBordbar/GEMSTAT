#include "ObjFunc.h"

double RMSEObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){

    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double squaredErr = 0.0;

    for(int i = 0;i<ground_truth.size();i++){
      double beta = 1.0;
      #ifdef BETAOPTTOGETHER
        if(NULL != par)
          beta = par->getBetaForSeq(i);
        squaredErr += least_square( prediction[i], ground_truth[i], beta, true );
      #else
        squaredErr += least_square( prediction[i], ground_truth[i], beta );
      #endif
  }

    double rmse = sqrt( squaredErr / ( nSeqs * nConds ) );
    return rmse;
}

double AvgCorrObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){

    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double totalSim = 0.0;

    for(int i = 0;i<ground_truth.size();i++){

      totalSim += corr(  prediction[i], ground_truth[i] );
  }

    return -totalSim/nSeqs;
  }

double PGPObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){

        assert(ground_truth.size() == prediction.size());
        int nSeqs = ground_truth.size();
        int nConds = ground_truth[0].size();
        double totalPGP = 0.0;

        for(int i = 0;i<ground_truth.size();i++){
          double beta = 1.0;
          #ifdef BETAOPTTOGETHER
        	beta = par->getBetaForSeq(i);
                totalPGP += pgp(  prediction[i], ground_truth[i], beta, true);
        	#else
        	totalPGP += pgp(  prediction[i], ground_truth[i], beta );
        	#endif
      }

      return totalPGP / nSeqs;
  }

double AvgCrossCorrObjFunc::exprSimCrossCorr( const vector< double >& x, const vector< double >& y )
  {
      vector< int > shifts;
      for ( int s = -maxShift; s <= maxShift; s++ )
      {
          shifts.push_back( s );
      }

      vector< double > cov;
      vector< double > corr;
      cross_corr( x, y, shifts, cov, corr );
      double result = 0, weightSum = 0;
      //     result = corr[maxShift];
      result = *max_element( corr.begin(), corr.end() );
      //     for ( int i = 0; i < shifts.size(); i++ ) {
      //         double weight = pow( shiftPenalty, abs( shifts[i] ) );
      //         weightSum += weight;
      //         result += weight * corr[i];
      //     }
      //     result /= weightSum;

      return result;
  }

double AvgCrossCorrObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){

    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double totalSim = 0.0;

    for(int i = 0;i<ground_truth.size();i++){
        totalSim += exprSimCrossCorr( prediction[i], ground_truth[i] );
    }

  return -totalSim / nSeqs;
  }


double LogisticRegressionObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){
    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double totalLL = 0.0;

    for(int i = 0;i<ground_truth.size();i++){
      vector<double> Y = ground_truth[i];
      vector<double> Ypred = prediction[i];

      double one_sequence_LL = 0.0;

      for(int j = 0;j<Y.size();j++){
        double one_gt = Y[i];
        double pred_prob = logistic(w*(Ypred[j] - bias));
        double singleLL = one_gt*log(pred_prob) + (1.0 - one_gt)*log(1.0 - pred_prob);
        one_sequence_LL += singleLL;
      }

      totalLL += one_sequence_LL;
    }
    return -totalLL;
}

RegularizedObjFunc::RegularizedObjFunc(ObjFunc* wrapped_obj_func, const ExprPar& centers, const ExprPar& l1, const ExprPar& l2)
{
  my_wrapped_obj_func = wrapped_obj_func;
  ExprPar tmp_energy_space = centers.my_factory->changeSpace(centers, ENERGY_SPACE);
  tmp_energy_space.getRawPars(my_centers );

  //It doesn't matter what space these are in, they are just storage for values.
  l1.getRawPars(lambda1 );
  l2.getRawPars(lambda2 );
  cache_pars = vector<double>(my_centers.size(),0.0);
  //cache_diffs(my_centers.size(),0.0);
  //cache_sq_diffs(my_centers.size(),0.0);
}

double RegularizedObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction, const ExprPar* par){

  double objective_value = my_wrapped_obj_func->eval( ground_truth, prediction, par );



  double l1_running_total = 0.0;
  double l2_running_total = 0.0;

  ExprPar tmp_energy_space = par->my_factory->changeSpace(*par, ENERGY_SPACE);
  tmp_energy_space.getRawPars(cache_pars );

  for(int i = 0;i<cache_pars.size();i++){
    double the_diff = abs(cache_pars[i] - my_centers[i]);
    l1_running_total += lambda1[i]*the_diff;
    l2_running_total += lambda2[i]*pow(the_diff,2.0);
  }

  objective_value += l1_running_total + l2_running_total;

  return objective_value;
}

double PeakWeightedObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){
    double on_threshold = 0.5;
    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double squaredErr = 0.0;

    for(int i = 0;i<ground_truth.size();i++){
      double beta = 1.0;
      #ifdef BETAOPTTOGETHER
        if(NULL != par)
          beta = par->getBetaForSeq(i);
        squaredErr += wted_least_square( prediction[i], ground_truth[i], beta, on_threshold, true );
      #else
        squaredErr += wted_least_square( prediction[i], ground_truth[i], beta, on_threshold );
      #endif
  }

    double rmse = sqrt( squaredErr / ( nSeqs * nConds ) );
    return rmse;
}

double Weighted_RMSEObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){
    #ifndef BETAOPTTOGETHER
        assert(false);
    #endif

    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double squaredErr = 0.0;

    for(int i = 0;i<ground_truth.size();i++){
      double beta = 1.0;
      if(NULL != par){ beta = par->getBetaForSeq(i); }

      for(int j = 0;j<nConds;j++){
          double single_sqr_error = (beta*prediction[i][j] - ground_truth[i][j]);
          single_sqr_error = weights->getElement(i,j)*pow(single_sqr_error,2);
          squaredErr += single_sqr_error;
      }
    }

    double rmse = sqrt( squaredErr / total_weight );
    return rmse;
}

void Weighted_ObjFunc_Mixin::set_weights(Matrix *in_weights){
    if(NULL != weights){delete weights;}
    weights = in_weights;

    //Caculate the total weight.
    total_weight = 0.0;
    for(int i = 0;i<weights->nRows();i++){
        for(int j = 0;j<weights->nCols();j++){
            total_weight+=weights->getElement(i,j);
        }
    }
}



double GroupedSoftMin_ObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){

    //FOR DEBUG
    cerr << "HELLO FROM THE grouped soft min objective" << endl;
    //cerr << " group mapping size is " << group_mapping.size() << endl;
    //cerr << " ground_truth  size is " << ground_truth.size() << endl;
    //cerr << " prediction    size is " << prediction.size() << endl;
    //cerr << " ground_truth  is " << ground_truth << endl;
    //cerr << " prediction is " << prediction << endl;

    assert(ground_truth.size() == prediction.size());

    vector< double > individual_scores(ground_truth.size(), 0.0);
    vector< double > group_scores(number_of_groups, 0.0);

    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();

    for(int i = 0;i<ground_truth.size();i++){
      double one_rmse = 0.0;
      double beta = 1.0;
      #ifdef BETAOPTTOGETHER
        if(NULL != par)
          beta = par->getBetaForSeq(i);
        one_rmse += least_square( prediction[i], ground_truth[i], beta, true );
      #else
        one_rmse += least_square( prediction[i], ground_truth[i], beta );
      #endif

        one_rmse = sqrt( one_rmse / nConds );
        individual_scores[i] = one_rmse;
    }

    for(int i = 0;i<individual_scores.size();i++){
      group_scores[group_mapping[i]] += exp(-5.0*individual_scores[i]);
    }

    for(int i = 0;i<group_scores.size();i++){
      group_scores[i] = -1.0*log(group_scores[i]);
    }

    double overall_score = 0.0;
    for(int i = 0;i<group_scores.size();i++){
      overall_score += group_scores[i];
    }

    return overall_score;
}

void GroupedSoftMin_ObjFunc::read_grouping_file(string filename){
  //parser is responsible for figuring out number of groups.
  //parser populates group_mapping.
  ifstream fin;
  fin.open(filename);
  if(fin){
    int group_nu;
    while(fin >> group_nu){
      group_mapping.push_back(group_nu);
    }
  }
  fin.close();
  int number_of_seqs;
  number_of_seqs = group_mapping.size();
  number_of_groups = group_mapping[number_of_seqs-1] + 1;
  //Temporary for example
  //cerr << " Hello from the grouped soft min file parser! you asked to read file " << filename << endl;
  //cerr << " read group mapping vector is" << group_mapping << endl;
  //cerr << " group mapping size is" << number_of_seqs << endl;
}


double Fold_Change_ObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){

  // TODO : 


    //FOR DEBUG
    //cerr << "HELLO FROM THE Fold_Change objective" << endl;
    //cerr << " group mapping size is " << group_mapping.size() << endl;
    //cerr << " ground_truth  size is " << ground_truth.size() << endl;
    //cerr << " prediction    size is " << prediction.size() << endl;
    //cerr << " ground_truth  is " << ground_truth << endl;
    //cerr << " prediction is " << prediction << endl;

    assert(ground_truth.size() == prediction.size());


    vector< double > individual_scores(ground_truth.size(), 0.0);
    vector< double > group_scores(number_of_groups, 0.0);
    //cerr << " number_of_groups "<< number_of_groups << endl;

    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    int nExp = ground_truth[0].size() / 2.0;

    for(int i = 0;i<ground_truth.size();i++){
      double one_rmse = 0.0;
      double beta = 1.0;
      vector< double > predicted_FoldChange(nExp, 0.0);
      vector< double > measured_FoldChange(nExp, 0.0);
      
      predicted_FoldChange = logFoldChange(prediction[i], treat_control_map);
      predicted_FoldChange = my_sigmoid(predicted_FoldChange);
      //cerr<<" predicted_FoldChange size " << predicted_FoldChange.size() << endl;
      //cerr<<" predicted_FoldChange " << predicted_FoldChange << endl;

      //cerr << "Before second log fold change " << endl;
      //cerr<<" measured_FoldChange size " << measured_FoldChange.size() << endl;
      measured_FoldChange  = logFoldChange_NA(ground_truth[i], treat_control_map, predicted_FoldChange);
      //cerr<<" measured_FoldChange " << measured_FoldChange << endl;
      #ifdef BETAOPTTOGETHER
        if(NULL != par)
          beta = par->getBetaForSeq(i);
        //cerr << "Before least_square beta "<< i << endl;
        one_rmse += least_square( predicted_FoldChange, measured_FoldChange, beta, true );
        //cerr << "one_rsme " << one_rmse << endl;
        //cerr << "after least_square beta "<< i << endl;
      #else
        //cerr << "Before least_square"<< i << endl;
        one_rmse += least_square( predicted_FoldChange, measured_FoldChange, beta );
        //cerr << "after least_square "<< i << endl;
      #endif

        //one_rmse = sqrt( one_rmse / nConds );
        individual_scores[i] = one_rmse;
    }
    //cerr << "individual_scores " << individual_scores << endl;

    for(int i = 0;i<individual_scores.size();i++){
      group_scores[group_mapping[i]] += exp(-10.0*individual_scores[i]);
    }
    //cerr << "group_scores " << group_scores << endl;

    //cerr << "after aggregation"<< endl;
    for(int i = 0;i<group_scores.size();i++){
      group_scores[i] = -1.0*log(group_scores[i]);
    }
    //cerr << "group_scores_minus_log " << group_scores << endl;
    //cerr << "after group_scores"<< endl;
    double overall_score = 0.0;
    for(int i = 0;i<group_scores.size();i++){
      overall_score += group_scores[i];
    }
    //cerr << "after overall_score"<< endl;
    //cerr << "overall_score " <<overall_score << endl;
    return overall_score;
}


void Fold_Change_ObjFunc::read_treat_control_file(string filename){
  // populates treat_control_map
  // 
  ifstream fin;
  fin.open(filename);
  if(fin){
    int trtcont_nu;
    while(fin >> trtcont_nu){
      treat_control_map.push_back(trtcont_nu);
    }
  }
  fin.close();
  //Temporary for example
  //cerr << " Hello from the Fold_Change_ObjFunc read_treat_control_file function! you asked to read file " << filename << endl;

  }

  double Weighted_Fold_Change_ObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){

    assert(ground_truth.size() == prediction.size());
        //FOR DEBUG
    //cerr << "HELLO FROM THE Weighted_Fold_Change_ObjFunc objective" << endl;


    vector< double > individual_scores(ground_truth.size(), 0.0);
    vector< double > group_scores(number_of_groups, 0.0);
    //cerr << " number_of_groups "<< number_of_groups << endl;

    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    int nExp = ground_truth[0].size() / 2.0;
    //double squaredErr = 0.0;

    for(int i = 0;i<ground_truth.size();i++){
      double one_rmse = 0.0;
      double beta = 1.0;
      vector< double > predicted_FoldChange(nExp, 0.0);
      vector< double > measured_FoldChange(nExp, 0.0);
      
      predicted_FoldChange = logFoldChange(prediction[i], treat_control_map);
      predicted_FoldChange = my_sigmoid(predicted_FoldChange);
      //cerr<<" predicted_FoldChange size " << predicted_FoldChange.size() << endl;
      //cerr<<" predicted_FoldChange " << predicted_FoldChange << endl;

      //cerr << "Before second log fold change " << endl;
      //cerr<<" measured_FoldChange size " << measured_FoldChange.size() << endl;
      measured_FoldChange  = logFoldChange_NA(ground_truth[i], treat_control_map, predicted_FoldChange);
      //cerr<<" measured_FoldChange " << measured_FoldChange << endl;
      #ifdef BETAOPTTOGETHER
        if(NULL != par)
          beta = par->getBetaForSeq(i);
        //cerr << "Before least_square beta "<< i << endl;
        for(int j = 0;j<nExp;j++){
          double single_sqr_error = (beta*predicted_FoldChange[j] - measured_FoldChange[j]);
          single_sqr_error = weights->getElement(i,(int) 2*j)*pow(single_sqr_error,2);
          //squaredErr += single_sqr_error;
          one_rmse += single_sqr_error;
        }
        // one_rmse += least_square( predicted_FoldChange, measured_FoldChange, beta, true );
        //cerr << "one_rsme " << one_rmse << endl;
        //cerr << "after least_square beta "<< i << endl;
      #else
        cout << "weights are not being considered, since BETAOPTTOGETHER was not defined" <<endl;
        one_rmse += least_square( predicted_FoldChange, measured_FoldChange, beta );
      #endif

        //one_rmse = sqrt( one_rmse / nConds );
        individual_scores[i] = one_rmse;
    }

    //cerr << "individual_scores " << individual_scores << endl;


    // ############## => change this to normal sum because in the first run I am only using one enhancer per gene and want to compare with the linear model
    // ##############
    // for(int i = 0;i<individual_scores.size();i++){
    //   group_scores[group_mapping[i]] += exp(-5.0*individual_scores[i]);
    // }
    // //cerr << "group_scores " << group_scores << endl;

    // //cerr << "after aggregation"<< endl;
    // for(int i = 0;i<group_scores.size();i++){
    //   group_scores[i] = -1.0*log(group_scores[i]);
    // }
    // ##############
    // ##############

    // This part of obj func is temporary and will be changed
    for(int i = 0;i<individual_scores.size();i++){
       group_scores[group_mapping[i]] += individual_scores[i];
    }


    //cerr << "group_scores_minus_log " << group_scores << endl;
    //cerr << "after group_scores"<< endl;
    double overall_score = 0.0;
    for(int i = 0;i<group_scores.size();i++){
      overall_score += group_scores[i];
    }
    //cerr << "after overall_score"<< endl;
    //cerr << "overall_score " <<overall_score << endl;
    return overall_score;

  }





